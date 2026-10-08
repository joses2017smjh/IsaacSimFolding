"""Check honest verdict wording and append behavior without creating documents."""
import os
from pathlib import Path
import tempfile
from publish_results import append_existing,result_block

def main():
    base={'source_summary_sha256':'a'*64,'manifest_sha256':'b'*64,'validation_errors':[],
          'result':{'counts':{'h10_settled':20,'h50_settled':10},'valid_pairs':48,
                    'expected_pairs':48,'h10_minus_h50_settled':10,
                    'exact_paired_one_sided_p':.02}}
    passing,identity=result_block(dict(base,classification='pass'))
    assert 'PASS on the eight development poses' in passing
    assert '20/48' in passing and '10/48' in passing
    assert 'held-out confirmation' in passing and 'unchanged baseline weights' in passing
    negative,_=result_block(dict(base,classification='negative'))
    assert 'No improvement under the preregistered development rule' in negative
    incomplete,_=result_block(dict(base,classification='incomplete',result=None,
                                   source_summary_sha256=None,validation_errors=['Missing source JSON']))
    assert 'no improvement claim' in incomplete and 'unavailable' in incomplete
    with tempfile.TemporaryDirectory(prefix='folding-publication-',dir='/tmp') as tmp:
        # Logs exercise the append algorithm; no test .md/.txt is created.
        path=Path(tmp)/'existing.log'
        path.write_text('Existing user content.\n')
        assert append_existing(path,passing,identity)=='appended'
        path.open('a').write('Concurrent editor addition.\n')
        assert append_existing(path,passing,identity)=='already_recorded'
        current=path.read_text()
        assert current.startswith('Existing user content.\n')
        assert current.endswith('Concurrent editor addition.\n')
        assert current.count(f'folding-runtime-result:{identity}:begin')==1
        missing=Path(tmp)/'missing.log'
        try:
            append_existing(missing,passing,identity)
        except FileNotFoundError:
            assert not missing.exists()
        else:
            raise AssertionError('Missing existing file was recreated')
    print('Publication checks passed: scoped verdicts, idempotent append, preservation of edits, no recreation of missing documents.')

if __name__=='__main__':
    main()
