import pandas as pd
import numpy as np
from itertools import combinations, product

class FuzzyRule:
    def __init__(self, lhs, rhs, count_full, count_lhs, count_rhs, n_samples):
        self.lhs = tuple(lhs)
        self.rhs = tuple(rhs)
        self.support = count_full / n_samples
        self.confidence = count_full / count_lhs if count_lhs > 0 else 0.0
        supp_rhs = count_rhs / n_samples 
        self.lift = self.confidence / supp_rhs if supp_rhs > 0 else 0

    def __repr__(self):
        return (
            f"{{{', '.join(self.lhs)}->{', '.join(self.rhs)}}} "
            f"(conf: {self.confidence:.3f}, supp: {self.support:.3f}, lift: {self.lift:.3f})"
        )

class Arim:

    def __init__(self, num_cols=None, min_supp=0.01, min_conf=0.5, max_rules=None, n_bins=5):
        self.min_supp = min_supp
        self.min_conf = min_conf 
        self.max_rules = max_rules
        self.n_bins = n_bins
        self.rules = None
        self.transactions = None

        self.num_cols = num_cols if num_cols is not None else []
        self.cat_cols = []
        self.centers = {}

        self.cat_modes = {}
        self.num_medians = {}
        self.known_cats = {}

    def calc_triang_memb(self, val, c_prev, c, c_next, is_first, is_last):
            if pd.isna(val):
                 return 0.0
            if is_first and val <= c:
                 return 1.0
            if is_last and val >= c:
                 return 1.0
            if c_prev is not None and c_prev <= val <= c:
                 return (val - c_prev) / (c - c_prev) if c != c_prev else 1.0
            if c_next is not None and c < val <= c_next:
                 return (c_next - val) / (c_next - c) if c_next != c else 1.0
            return 0.0

    def fuzzify_row(self, row):
        fuzz_dict = {}
        for col in self.num_cols:
            val = row.get(col, np.nan)
            centers = self.centers[col]
            k = len(centers)
            for i in range(k):
                label = f'bin_{i+1}'
                c_prev = centers[i-1] if i > 0 else None
                c = centers[i]
                c_next = centers[i+1] if i < k-1 else None 
                mu = self.calc_triang_memb(val, c_prev, c, c_next, i==0, i==k-1)
                fuzz_dict[f'{col}={label}'] = mu
        for col in self.cat_cols:
             val = row.get(col, np.nan)
             for cat in self.known_cats.get(col, []):
                  fuzz_dict[f'{col}={cat}'] = 1.0 if str(val) == str(cat) else 0.0
        return fuzz_dict

    def build_fuzzy_matrix(self, data):
         rows = [self.fuzzify_row(row) for _, row in data.iterrows()]
         return pd.DataFrame(rows, index = data.index)

    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data should be DataFrame!")
        self.cat_cols = [cat for cat in data.columns if cat not in self.num_cols]
        for col in self.num_cols:
            if col in data.columns:
                u = data[col].dropna()
                u_min = u.min() if not u.empty else 0.0
                u_max = u.max() if not u.empty else 1.0
                if u_min == u_max:
                     u.max += 1
                self.centers[col] = np.linspace(u_min, u_max, self.n_bins)
                self.num_medians[col] = u.median() if not u.empty else 0.0

        for col in self.cat_cols:
             if col in data.columns:
                  u = data[col].dropna()
                  self.known_cats[col] = u.unique().tolist()
                  self.cat_modes[col] = u.mode().iloc[0] if not u.empty else None

        fuzzy_df = self.build_fuzzy_matrix(data)
        n_samples = len(fuzzy_df)
        feature_to_terms = {}
        for col in fuzzy_df.columns:
            feat = col.split('=', 1)[0]
            feature_to_terms.setdefault(feat, []).append(col)
        features = list(feature_to_terms.keys())
        single_counts = {col: fuzzy_df[col].sum() for col in fuzzy_df.columns}
        frequent_single = {c: cnt for c, cnt in single_counts.items() if (cnt / n_samples) >= self.min_supp}

        rules = []
        for target_feat in features:
             for rhs_term in feature_to_terms[target_feat]:
                if rhs_term not in frequent_single:
                    continue
                count_rhs = frequent_single[rhs_term]
                mu_rhs = fuzzy_df[rhs_term].to_numpy()
                available_feats = [feat for feat in features if feat != target_feat ]
                frequent_lhs = {0: [(tuple(), np.ones(n_samples), float(n_samples))]}

                for k in range(1, len(available_feats)+1):
                    frequent_lhs[k] = []
                    for feat_comb in combinations(available_feats, k):
                         for ant_tuple in product(*[feature_to_terms[f] for f in feat_comb]):
                                if k > 1:
                                    subsets_ok = all(any(sub == prev for prev, _, _ in frequent_lhs[k-1]) for sub in combinations(ant_tuple, k-1))
                                    if not subsets_ok:
                                        continue
                                mu_lhs = np.prod([fuzzy_df[t].to_numpy() for t in ant_tuple], axis=0)
                                count_lhs = mu_lhs.sum()
                                if (count_lhs/n_samples) < self.min_supp:
                                     continue
                                frequent_lhs[k].append((ant_tuple, mu_lhs, count_lhs))
                                mu_full = mu_lhs * mu_rhs
                                count_full = mu_full.sum()
                                supp_full = count_full / n_samples
                                if supp_full >= self.min_supp:
                                    conf = count_full / count_lhs if count_lhs > 0 else 0.0
                                    if conf >= self.min_conf:
                                        rules.append(FuzzyRule(ant_tuple, (rhs_term, ), count_full, count_lhs, count_rhs, n_samples))
        rules.sort(key=lambda x: (x.confidence, x.lift, x.support), reverse=True)
        self.rules = rules[:self.max_rules] if self.max_rules else rules
        return self

    def match_score(self, row_fuzzy, full_cols, rule):
        mu_vals = []
        for item in rule.lhs:
            col, _ = item.split('=', 1)
            if col not in full_cols:
                return -1.0
            mu = row_fuzzy.get(item, 0.0)
            if mu == 0.0:
                return -1.0
            mu_vals.append(mu)
        if not mu_vals:
            return -1.0
        t_norm_lhs = np.prod(mu_vals)
        return (t_norm_lhs**2) * rule.confidence * np.log(1 + len(rule.lhs))

    def decode_term_value(self, term_str):
        col, label = term_str.split('=', 1)
        if col in self.num_cols:
            idx = int(label.replace('bin_','')) - 1
            return self.centers[col][idx]
        else:
            return label

    def transform(self, data):
        if self.rules is None:
            raise ValueError("Rules not found. Call fit() first.")
        result = data.copy()
        for idx, row in result.iterrows():
            missing = [c for c in result.columns if pd.isna(row[c])]
            if not missing:
                continue
            full = [c for c in result.columns if c not in missing]
            row_fuzzy = self.fuzzify_row(row)
            for missing_col in missing:
                candidates = [r for r in self.rules if r.rhs[0].split('=', 1)[0] == missing_col]
                best_rule = None
                best_score = -1.0
                for r in candidates:
                    score = self.match_score(row_fuzzy, full, r)
                    if score > best_score:
                        best_score = score
                        best_rule = r
                if best_rule and best_score > 0:
                    result.at[idx, missing_col] = self.decode_term_value(best_rule.rhs[0])
                else:
                    if missing_col in self.cat_cols:
                        result.at[idx, missing_col] = self.cat_modes.get(missing_col, None)
                    else:
                        result.at[idx, missing_col] = self.num_medians.get(missing_col, 0.0)
        return result

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


            


                            
                              


        
                 
                 

    
    
    
        