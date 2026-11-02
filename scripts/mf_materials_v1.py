# -*- coding: utf-8 -*-
"""MF-v1 材料：24 个算法任务 × 4 种风格（中文说明 / 英文说明 / 英文论文 / Python 代码），同一任务的四种风格共用同一组标识符与数字。
槽位：{L} 列表名、{R} 结果名、{N} 小整数。每个任务按确定性规则生成 5 组填充。中文里的标识符前后留空格（中文技术写作常见），
使「 items」「 total」等 token 在四种风格中相同，可作为跨风格共用 token 的读出位置。"""
from __future__ import annotations

import hashlib

STYLES = ['zh', 'en', 'paper', 'code']
LISTS = ['items', 'values', 'scores', 'prices', 'nums', 'data', 'weights', 'counts']
RESULTS = ['total', 'result', 'out', 'acc', 'best', 'answer', 'value', 'res']

T = {
    'sum': {
        'zh': '我们要把列表 {L} 里的所有数字加起来。先令 {R} 等于 {N}，然后依次取出 {L} 中的每个元素，把它加到 {R} 上。全部加完以后，{R} 就是总和。',
        'en': 'We want to add up all the numbers in the list {L}. First set {R} to {N}, then take each element of {L} in turn and add it to {R}. When the loop ends, {R} holds the sum.',
        'paper': 'Let {L} denote a finite sequence of reals. We define the aggregate {R} = {N} + \\sum_i {L}_i and compute it with a single linear pass, which requires O(n) time and O(1) additional memory.',
        'code': 'def add_all({L}):\n    {R} = {N}\n    for x in {L}:\n        {R} += x\n    return {R}\n',
    },
    'max': {
        'zh': '要找出列表 {L} 中最大的数。先把 {R} 设为 {L} 的第一个元素，再逐个比较其余元素；如果某个元素比 {R} 大，就用它替换 {R}。最后 {R} 就是最大值。',
        'en': 'To find the largest number in the list {L}, first set {R} to the first element of {L}. Then compare every other element with {R}; whenever an element is larger, replace {R} with it. At the end, {R} is the maximum.',
        'paper': 'Given a sequence {L} of length n, the maximum {R} = \\max_i {L}_i is obtained by a single scan that maintains the running maximum; the scan performs exactly n - 1 comparisons.',
        'code': 'def largest({L}):\n    {R} = {L}[0]\n    for x in {L}[1:]:\n        if x > {R}:\n            {R} = x\n    return {R}\n',
    },
    'count_gt': {
        'zh': '统计列表 {L} 中有多少个数大于 {N}。先令 {R} 为 0，然后检查 {L} 的每个元素，只要它大于 {N}，就把 {R} 加 1。检查完毕后，{R} 就是要求的个数。',
        'en': 'Count how many numbers in the list {L} are greater than {N}. Start with {R} equal to 0, check each element of {L}, and add 1 to {R} whenever the element is greater than {N}. Afterwards, {R} is the count.',
        'paper': 'For a threshold {N}, we define the exceedance count {R} = |\\{ i : {L}_i > {N} \\}|, i.e., the number of entries of {L} that strictly exceed the threshold. It is computed in O(n) time.',
        'code': 'def count_above({L}):\n    {R} = 0\n    for x in {L}:\n        if x > {N}:\n            {R} += 1\n    return {R}\n',
    },
    'reverse': {
        'zh': '把列表 {L} 的顺序反过来。先准备一个空列表 {R}，然后从 {L} 的最后一个元素开始，依次向前取出元素并放进 {R}。这样 {R} 就是反转后的列表。',
        'en': 'Reverse the order of the list {L}. Prepare an empty list {R}, then start from the last element of {L} and move backwards, appending each element to {R}. In the end, {R} is the reversed list.',
        'paper': 'The reversal operator maps a sequence {L} = ({L}_1, ..., {L}_n) to {R} = ({L}_n, ..., {L}_1). We construct {R} by iterating over indices in descending order, which takes linear time.',
        'code': 'def reverse_list({L}):\n    {R} = []\n    for i in range(len({L}) - 1, -1, -1):\n        {R}.append({L}[i])\n    return {R}\n',
    },
    'mean': {
        'zh': '计算列表 {L} 的平均值。先把 {L} 中所有元素加起来得到总和，再除以元素的个数，结果记为 {R}。如果 {L} 是空的，就让 {R} 等于 {N}。',
        'en': 'Compute the average of the list {L}. Add all the elements of {L} to get the total, then divide by the number of elements and call the result {R}. If {L} is empty, let {R} be {N}.',
        'paper': 'The empirical mean of a sequence {L} of length n is {R} = \\frac{1}{n} \\sum_i {L}_i. For the degenerate case n = 0 we adopt the convention {R} = {N}.',
        'code': 'def mean({L}):\n    if not {L}:\n        return {N}\n    {R} = sum({L}) / len({L})\n    return {R}\n',
    },
    'evens': {
        'zh': '从列表 {L} 中挑出所有偶数。准备一个空列表 {R}，逐个检查 {L} 的元素，如果它除以 2 的余数是 0，就把它放进 {R}。最后 {R} 里只剩偶数。',
        'en': 'Select all the even numbers from the list {L}. Prepare an empty list {R}, check each element of {L}, and if its remainder when divided by 2 is 0, put it into {R}. Finally, {R} contains only even numbers.',
        'paper': 'We define the even subsequence {R} = ({L}_i : {L}_i \\equiv 0 \\pmod 2), preserving the original order. Membership is decided by a constant-time parity test on each entry.',
        'code': 'def evens({L}):\n    {R} = []\n    for x in {L}:\n        if x % 2 == 0:\n            {R}.append(x)\n    return {R}\n',
    },
    'index_of': {
        'zh': '在列表 {L} 中寻找数字 {N} 第一次出现的位置。从下标 0 开始逐个比较，一旦某个元素等于 {N}，就把它的下标记为 {R} 并停止；如果找不到，{R} 为 -1。',
        'en': 'Find the position where the number {N} first appears in the list {L}. Compare elements one by one starting from index 0; as soon as an element equals {N}, record its index as {R} and stop. If it is never found, {R} is -1.',
        'paper': 'The first-occurrence index is {R} = \\min \\{ i : {L}_i = {N} \\}, with the convention {R} = -1 when the set is empty. A left-to-right scan terminates at the first match.',
        'code': 'def find({L}):\n    {R} = -1\n    for i, x in enumerate({L}):\n        if x == {N}:\n            {R} = i\n            break\n    return {R}\n',
    },
    'word_count': {
        'zh': '统计字符串 {L} 中有多少个单词。先按空格把 {L} 切分成若干片段，去掉空的片段，剩下的片段个数就是单词数，记为 {R}。',
        'en': 'Count how many words the string {L} contains. Split {L} on spaces into pieces, discard the empty pieces, and the number of remaining pieces is the word count, called {R}.',
        'paper': 'We tokenize the string {L} on whitespace and define {R} as the number of non-empty tokens. This simple tokenizer serves as the baseline in all experiments reported in Section {N}.',
        'code': 'def word_count({L}):\n    {R} = 0\n    for w in {L}.split(" "):\n        if w:\n            {R} += 1\n    return {R}\n',
    },
    'palindrome': {
        'zh': '判断字符串 {L} 是不是回文。用两个下标分别从开头和结尾向中间移动，每次比较两个字符；只要有一处不同，{R} 就是假，否则 {R} 是真。',
        'en': 'Decide whether the string {L} is a palindrome. Move two indices from the beginning and the end towards the middle, comparing characters at each step; if any pair differs, {R} is false, otherwise {R} is true.',
        'paper': 'A string {L} of length n is a palindrome iff {L}_i = {L}_{n+1-i} for all i. The indicator {R} is evaluated with two pointers in at most \\lfloor n/2 \\rfloor comparisons.',
        'code': 'def is_palindrome({L}):\n    i, j = 0, len({L}) - 1\n    {R} = True\n    while i < j:\n        if {L}[i] != {L}[j]:\n            {R} = False\n        i, j = i + 1, j - 1\n    return {R}\n',
    },
    'factorial': {
        'zh': '计算 {N} 的阶乘。先令 {R} 等于 1，然后让 i 从 1 走到 {N}，每一步都把 {R} 乘以 i。循环结束后，{R} 就是 {N} 的阶乘。',
        'en': 'Compute the factorial of {N}. Start with {R} equal to 1, then let i run from 1 to {N}, multiplying {R} by i at every step. After the loop, {R} is the factorial of {N}.',
        'paper': 'The factorial is defined recursively by 0! = 1 and n! = n (n - 1)!. We evaluate {R} = {N}! iteratively, which requires {N} multiplications and no auxiliary storage.',
        'code': 'def factorial():\n    {R} = 1\n    for i in range(1, {N} + 1):\n        {R} *= i\n    return {R}\n',
    },
    'fib': {
        'zh': '生成斐波那契数列的前 {N} 项，存进列表 {R}。前两项是 0 和 1，之后每一项都等于它前面两项的和，一直生成到 {R} 里有 {N} 个数为止。',
        'en': 'Generate the first {N} Fibonacci numbers and store them in the list {R}. The first two terms are 0 and 1, and every later term is the sum of the two terms before it, until {R} holds {N} numbers.',
        'paper': 'The Fibonacci sequence satisfies F_0 = 0, F_1 = 1 and F_k = F_{k-1} + F_{k-2}. We materialize the prefix {R} = (F_0, ..., F_{{N}-1}) in O({N}) time.',
        'code': 'def fib():\n    {R} = [0, 1]\n    while len({R}) < {N}:\n        {R}.append({R}[-1] + {R}[-2])\n    return {R}[:{N}]\n',
    },
    'dedup': {
        'zh': '去掉列表 {L} 中重复的元素，并保持原来的顺序。用一个集合记录已经见过的元素，逐个检查 {L}，没见过的就放进结果列表 {R}，同时记入集合。',
        'en': 'Remove duplicate elements from the list {L} while keeping the original order. Keep a set of elements already seen, walk through {L}, and put each unseen element into the result list {R} while adding it to the set.',
        'paper': 'Order-preserving deduplication returns the subsequence {R} of first occurrences in {L}. Using a hash set, the expected running time is O(n) and the memory overhead is O(k) for k distinct values.',
        'code': 'def dedup({L}):\n    seen = set()\n    {R} = []\n    for x in {L}:\n        if x not in seen:\n            seen.add(x)\n            {R}.append(x)\n    return {R}\n',
    },
    'sort': {
        'zh': '把列表 {L} 从小到大排序。每一轮从剩下的元素里找出最小的一个，放到结果列表 {R} 的末尾，再把它从剩下的元素中删掉，直到没有元素为止。',
        'en': 'Sort the list {L} in ascending order. In each round, find the smallest of the remaining elements, append it to the result list {R}, and remove it from the remaining elements, until nothing is left.',
        'paper': 'Selection sort produces the ordered sequence {R} by repeatedly extracting the minimum of the unsorted remainder of {L}. It performs \\Theta(n^2) comparisons regardless of the input order.',
        'code': 'def selection_sort({L}):\n    rest = list({L})\n    {R} = []\n    while rest:\n        m = min(rest)\n        {R}.append(m)\n        rest.remove(m)\n    return {R}\n',
    },
    'product': {
        'zh': '求列表 {L} 中所有数的乘积。先令 {R} 等于 1，然后把 {L} 的每个元素依次乘到 {R} 上。乘完以后，{R} 就是这些数的乘积。',
        'en': 'Compute the product of all the numbers in the list {L}. Set {R} to 1, then multiply {R} by each element of {L} in turn. When every element has been used, {R} is the product.',
        'paper': 'We define the multiplicative aggregate {R} = \\prod_i {L}_i, with the empty product equal to 1. Overflow is avoided in practice by accumulating \\log |{L}_i| instead.',
        'code': 'def product({L}):\n    {R} = 1\n    for x in {L}:\n        {R} *= x\n    return {R}\n',
    },
    'min': {
        'zh': '要找出列表 {L} 中最小的数。先把 {R} 设为一个很大的数，再逐个检查 {L} 的元素；如果某个元素比 {R} 小，就用它替换 {R}。检查完毕后 {R} 就是最小值。',
        'en': 'To find the smallest number in the list {L}, set {R} to a very large number first. Then check each element of {L}; whenever an element is smaller than {R}, replace {R} with it. After the check, {R} is the minimum.',
        'paper': 'The minimum {R} = \\min_i {L}_i is computed by a single pass initialized at +\\infty. The same pass can return the arg-min at no additional asymptotic cost.',
        'code': 'def smallest({L}):\n    {R} = float("inf")\n    for x in {L}:\n        if x < {R}:\n            {R} = x\n    return {R}\n',
    },
    'squares': {
        'zh': '把列表 {L} 中的每个数平方，结果放进新列表 {R}。逐个取出 {L} 的元素，计算它乘以自己的值，再把这个值追加到 {R} 的末尾。',
        'en': 'Square every number in the list {L} and put the results into a new list {R}. Take each element of {L}, multiply it by itself, and append that value to the end of {R}.',
        'paper': 'The element-wise square map sends {L} to {R} with {R}_i = {L}_i^2. It is a special case of the power map x \\mapsto x^p with p = 2 and preserves the length of the sequence.',
        'code': 'def squares({L}):\n    {R} = []\n    for x in {L}:\n        {R}.append(x * x)\n    return {R}\n',
    },
    'contains': {
        'zh': '判断数字 {N} 是否在列表 {L} 中。先令 {R} 为假，然后逐个比较 {L} 的元素，只要有一个等于 {N}，就把 {R} 设为真并停止检查。',
        'en': 'Check whether the number {N} occurs in the list {L}. Let {R} be false at first, then compare the elements of {L} one by one; as soon as one equals {N}, set {R} to true and stop checking.',
        'paper': 'The membership indicator {R} = 1[{N} \\in {L}] is evaluated by linear search. For unsorted input this is optimal, whereas a sorted {L} admits binary search in O(\\log n).',
        'code': 'def contains({L}):\n    {R} = False\n    for x in {L}:\n        if x == {N}:\n            {R} = True\n            break\n    return {R}\n',
    },
    'merge': {
        'zh': '把两个已经排好序的列表 {L} 和 {R} 合并成一个有序列表。每次比较两个列表开头的元素，把较小的那个取出来放进结果，直到两个列表都取完。',
        'en': 'Merge two sorted lists {L} and {R} into one sorted list. Each time, compare the first elements of the two lists, move the smaller one into the output, and repeat until both lists are empty.',
        'paper': 'Given sorted sequences {L} and {R}, the merge step of merge sort produces their sorted union in |{L}| + |{R}| - 1 comparisons in the worst case, which is optimal.',
        'code': 'def merge({L}, {R}):\n    out, i, j = [], 0, 0\n    while i < len({L}) and j < len({R}):\n        if {L}[i] <= {R}[j]:\n            out.append({L}[i]); i += 1\n        else:\n            out.append({R}[j]); j += 1\n    return out + {L}[i:] + {R}[j:]\n',
    },
    'cumsum': {
        'zh': '计算列表 {L} 的前缀和。用 {R} 保存结果，用一个变量记录到目前为止的累加值；每读到 {L} 的一个元素，就把它加进累加值，再把累加值追加到 {R}。',
        'en': 'Compute the prefix sums of the list {L}. Store the result in {R} and keep a running total; each time an element of {L} is read, add it to the running total and append the running total to {R}.',
        'paper': 'The prefix-sum transform maps {L} to {R} with {R}_k = \\sum_{i \\le k} {L}_i. It underlies constant-time range queries, since \\sum_{a < i \\le b} {L}_i = {R}_b - {R}_a.',
        'code': 'def prefix_sums({L}):\n    {R} = []\n    running = 0\n    for x in {L}:\n        running += x\n        {R}.append(running)\n    return {R}\n',
    },
    'count_eq': {
        'zh': '统计数字 {N} 在列表 {L} 中出现了几次。先令 {R} 为 0，然后逐个检查 {L} 的元素，每遇到一个等于 {N} 的元素，就把 {R} 加 1。',
        'en': 'Count how many times the number {N} appears in the list {L}. Set {R} to 0, then examine the elements of {L} one at a time, adding 1 to {R} for every element equal to {N}.',
        'paper': 'The frequency of a value {N} in {L} is {R} = \\sum_i 1[{L}_i = {N}]. Normalizing {R} by n yields the empirical probability mass at {N}.',
        'code': 'def frequency({L}):\n    {R} = 0\n    for x in {L}:\n        if x == {N}:\n            {R} += 1\n    return {R}\n',
    },
    'swap': {
        'zh': '交换两个变量 {L} 和 {R} 的值。先用一个临时变量保存 {L} 的值，再把 {R} 的值赋给 {L}，最后把临时变量的值赋给 {R}。',
        'en': 'Swap the values of the two variables {L} and {R}. First save the value of {L} in a temporary variable, then assign the value of {R} to {L}, and finally assign the saved value to {R}.',
        'paper': 'A transposition exchanges the contents of two registers {L} and {R}. With one auxiliary register the exchange takes three assignments; the XOR trick removes the auxiliary register for integers.',
        'code': 'def swap({L}, {R}):\n    tmp = {L}\n    {L} = {R}\n    {R} = tmp\n    return {L}, {R}\n',
    },
    'clamp': {
        'zh': '把列表 {L} 中所有大于 {N} 的数都改成 {N}，结果放进新列表 {R}。逐个检查 {L} 的元素，大于 {N} 的放入 {N}，其余的原样放入。',
        'en': 'Replace every number in the list {L} that is greater than {N} with {N}, putting the results into a new list {R}. Check each element of {L}; put {N} if it is greater than {N}, otherwise put the element itself.',
        'paper': 'The clipping operator with ceiling {N} maps each entry to {R}_i = \\min({L}_i, {N}). Clipping is non-expansive and is widely used to bound gradient magnitudes during training.',
        'code': 'def clip({L}):\n    {R} = []\n    for x in {L}:\n        {R}.append(min(x, {N}))\n    return {R}\n',
    },
    'gcd': {
        'zh': '求两个正整数 {L} 和 {R} 的最大公约数。只要 {R} 不等于 0，就用 {L} 除以 {R} 的余数替换 {R}，同时把原来的 {R} 赋给 {L}；最后的 {L} 就是答案。',
        'en': 'Find the greatest common divisor of two positive integers {L} and {R}. While {R} is not 0, replace {R} with the remainder of {L} divided by {R} and give the old {R} to {L}. The final {L} is the answer.',
        'paper': 'Euclid\'s algorithm computes \\gcd({L}, {R}) through the recurrence \\gcd(a, b) = \\gcd(b, a \\bmod b). By Lame\'s theorem the number of steps is O(\\log \\min({L}, {R})).',
        'code': 'def gcd({L}, {R}):\n    while {R} != 0:\n        {L}, {R} = {R}, {L} % {R}\n    return {L}\n',
    },
    'is_prime': {
        'zh': '判断 {N} 是不是质数。如果 {N} 小于 2，{R} 为假；否则让 i 从 2 开始试除，一直试到 i 的平方超过 {N}，只要有一次能整除，{R} 就是假。',
        'en': 'Decide whether {N} is a prime number. If {N} is less than 2, {R} is false; otherwise try dividing by i starting from 2 until i squared exceeds {N}, and if any division is exact, {R} is false.',
        'paper': 'Trial division certifies primality of {N} by testing all candidate divisors 2 \\le i \\le \\sqrt{{N}}. The indicator {R} therefore costs O(\\sqrt{{N}}) divisions in the worst case.',
        'code': 'def is_prime():\n    if {N} < 2:\n        return False\n    {R} = True\n    i = 2\n    while i * i <= {N}:\n        if {N} % i == 0:\n            {R} = False\n        i += 1\n    return {R}\n',
    },
}


def fillers(task: str, n: int = 5):
    out = []
    for j in range(n):
        h = hashlib.sha256(f'MF-v1|{task}|{j}'.encode()).digest()
        L = LISTS[h[0] % len(LISTS)]
        R = RESULTS[h[1] % len(RESULTS)]
        out.append({'L': L, 'R': R, 'N': str(2 + h[2] % 8)})
    return out


def render(task: str, style: str, f) -> str:
    return T[task][style].replace('{L}', f['L']).replace('{R}', f['R']).replace('{N}', f['N'])


def split():
    """按任务确定性切分：8 个确认任务，其余为发现任务（同一任务的四种风格与全部填充在同一划分）。"""
    tasks = sorted(T, key=lambda t: hashlib.sha256(f'MF-v1|split|{t}'.encode()).hexdigest())
    return {'conf': sorted(tasks[:8]), 'disc': sorted(tasks[8:])}


def corpus():
    """[{'id', 'task', 'style', 'split', 'fill', 'text'}]"""
    sp = split()
    out = []
    for s, tasks in sp.items():
        for t in tasks:
            for j, f in enumerate(fillers(t)):
                for st in STYLES:
                    out.append({'id': f'{t}/{j}/{st}', 'task': t, 'item': f'{t}/{j}', 'style': st, 'split': s, 'fill': f, 'text': render(t, st, f)})
    return out


# ---------------- MF-v5：自然切换材料（只用确认集任务）----------------
CLOSING = {'zh': '以上代码只遍历一次输入，时间复杂度是线性的。', 'en': 'The code above makes a single pass over the input, so it runs in linear time.'}


def _first_sentence(text: str, lang: str) -> str:
    cut = text.find('。') + 1 if lang == 'zh' else text.find('. ') + 1
    return text[:cut] if cut > 0 else text


def _last_sentence(text: str, lang: str) -> str:
    t = text.rstrip()
    if lang == 'zh':
        parts = [p for p in t.split('。') if p]
        return parts[-1] + '。'
    parts = [p for p in t.split('. ') if p]
    return parts[-1] if parts[-1].endswith('.') else parts[-1] + '.'


def mixed_corpus():
    """[{'id', 'task', 'kind', 'lang', 'text', 'spans': [(start, end, segment)]}]，segment ∈ prose / fence / code / comment。
    md：说明段落 + ```python 代码块 ``` + 结语；cmt：代码在函数头之后与 return 之前各插一行注释（取说明的首句与末句）。"""
    out = []
    for t in split()['conf']:
        for j, f in enumerate(fillers(t)):
            code = render(t, 'code', f)
            for lang in ('zh', 'en'):
                prose = render(t, lang, f)
                pieces = [(prose + '\n\n', 'prose'), ('```python\n', 'fence'), (code, 'code'), ('```\n\n', 'fence'), (CLOSING[lang], 'prose')]
                text, spans = '', []
                for s, seg in pieces:
                    spans.append((len(text), len(text) + len(s), seg))
                    text += s
                out.append({'id': f'{t}/{j}/md/{lang}', 'task': t, 'kind': 'md', 'lang': lang, 'text': text, 'spans': spans})
                lines = code.split('\n')
                ret = max(i for i, ln in enumerate(lines) if ln.strip().startswith('return'))
                c1 = '    # ' + _first_sentence(prose, lang) + '\n'
                c2 = '    # ' + _last_sentence(prose, lang) + '\n'
                pieces = [(lines[0] + '\n', 'code'), (c1, 'comment'), ('\n'.join(lines[1:ret]) + '\n', 'code'), (c2, 'comment'),
                          ('\n'.join(lines[ret:]), 'code')]
                text, spans = '', []
                for s, seg in pieces:
                    spans.append((len(text), len(text) + len(s), seg))
                    text += s
                out.append({'id': f'{t}/{j}/cmt/{lang}', 'task': t, 'kind': 'cmt', 'lang': lang, 'text': text, 'spans': spans})
    return out
