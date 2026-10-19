"""Semantic-key negative fixtures; no empirical reproduction is certified."""
import csv,importlib.util,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('strict_v2',Path(__file__).with_name('strict_csv_schema_v2.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class SemanticKeyTests(unittest.TestCase):
    def compare(self,rel,mutation=None):
        keys=list(s.SEMANTIC_INPUT_KEYS[rel]);header=keys+['y_anemia','y_stunting','diagnostic_pass']
        rows=[{'ubigeo':'000001','year':'2019','universe':'FIXTURE_A','y_anemia':'.2','y_stunting':'.1','diagnostic_pass':'True'},
              {'ubigeo':'000001','year':'2019','universe':'FIXTURE_B','y_anemia':'.3','y_stunting':'.2','diagnostic_pass':'True'}]
        rows=[{k:r[k]for k in header}for r in rows];fresh=[r.copy()for r in rows];fh=header[:]
        if mutation:fh,fresh,header,rows=mutation(fh,fresh,header,rows)
        with tempfile.TemporaryDirectory()as tmp:
            a,b=Path(tmp)/'fresh.csv',Path(tmp)/'reference.csv'
            for p,h,rs in ((a,fh,fresh),(b,header,rows)):
                with p.open('w',newline='')as stream:
                    writer=csv.DictWriter(stream,fieldnames=h);writer.writeheader();writer.writerows(rs)
            return s.compare_file(a,b,1e-10,0.,rel)
    def all_paths(self,mutation,expected):
        for rel in s.SEMANTIC_INPUT_KEYS:
            with self.subTest(path=rel):self.assertEqual(self.compare(rel,mutation)['status'],expected)
    def test_district_repeats_with_different_universe_are_valid(self):self.all_paths(None,'PASS')
    def test_same_semantic_composite_duplicate_still_fails(self):
        def m(fh,f,h,r):return fh,[f[0],f[0].copy()],h,[r[0],r[0].copy()]
        self.all_paths(m,'FAIL_DUPLICATE_COMPARISON_KEY')
    def test_dropped_semantic_dimension_on_both_sides_fails(self):
        def m(fh,f,h,r):
            fh.remove('universe');h.remove('universe')
            for row in f+r:row.pop('universe')
            return fh,f,h,r
        self.all_paths(m,'FAIL_SCHEMA_COMPARISON_KEY')
    def test_dropped_endpoint_field_fails(self):
        def m(fh,f,h,r):
            fh.remove('y_stunting')
            for row in f:row.pop('y_stunting')
            return fh,f,h,r
        self.all_paths(m,'FAIL_SCHEMA')
    def test_extra_statistical_field_fails(self):
        def m(fh,f,h,r):
            fh.append('undeclared_statistic')
            for row in f:row['undeclared_statistic']='.5'
            return fh,f,h,r
        self.all_paths(m,'FAIL_SCHEMA')
    def test_same_cohort_reordered_input_fails(self):self.all_paths(lambda fh,f,h,r:(fh,f[::-1],h,r),'FAIL_COHORT_ROW_ORDER')
    def test_dropped_cohort_row_fails(self):self.all_paths(lambda fh,f,h,r:(fh,f[:1],h,r),'FAIL_COHORT_ROW_COUNT')
    def test_changed_cohort_key_fails(self):
        def m(fh,f,h,r):f[0]['universe']='FIXTURE_UNPLANNED';return fh,f,h,r
        self.all_paths(m,'FAIL_COHORT_KEY_UNIVERSE')
    def test_numeric_difference_is_not_masked(self):
        def m(fh,f,h,r):f[0]['y_anemia']='.21';return fh,f,h,r
        self.all_paths(m,'FAIL_VALUES')
    def test_boolean_diagnostic_difference_is_not_masked(self):
        def m(fh,f,h,r):f[0]['diagnostic_pass']='False';return fh,f,h,r
        self.all_paths(m,'FAIL_VALUES')
    def test_statistical_column_order_change_fails(self):self.all_paths(lambda fh,f,h,r:(fh[::-1],f,h,r),'FAIL_SCHEMA_COLUMN_ORDER')
    def test_missing_numeric_cell_is_not_masked(self):
        def m(fh,f,h,r):f[0]['y_stunting']='';return fh,f,h,r
        self.all_paths(m,'FAIL_VALUES')

if __name__=='__main__':unittest.main()
