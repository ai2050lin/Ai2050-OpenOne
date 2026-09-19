memo = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
sec = r'D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/_p2852_memo_section.md'
body = open(sec, encoding='utf-8').read()
with open(memo, 'a', encoding='utf-8') as f:
    f.write(body)
lines = open(memo, encoding='utf-8').read().splitlines()
hits = [i + 1 for i, l in enumerate(lines) if 'Phase 2852' in l]
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2852_memo_append.txt'
open(rep, 'w', encoding='utf-8').write(
    'total_lines=%d\np2852_at_lines=%s\ntail=%s\n' % (
        len(lines), hits, lines[-1][:80]))
print('append done')
