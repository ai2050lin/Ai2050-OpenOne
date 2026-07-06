# Phase 204 global trajectory stop-execution atlas

## Trajectory Summary
| model | relation | pair | mode | protocol | rows | stable | drift | period | continued after period | eos ended | avg stop chain |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | color | en->en | natural | plain | 12 | 0 | 12 | 8 | 6 | 0 | 1.50 |
| qwen3 | color | en->en | natural | short_answer | 12 | 0 | 12 | 9 | 9 | 0 | 1.17 |
| qwen3 | color | en->en | natural | stop_explicit | 12 | 0 | 12 | 10 | 10 | 0 | 1.00 |
| qwen3 | color | en->en | post_answer | plain | 12 | 0 | 12 | 8 | 7 | 0 | 2.00 |
| qwen3 | color | en->en | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.17 |
| qwen3 | color | en->en | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| qwen3 | color | en->zh | natural | plain | 12 | 0 | 9 | 3 | 3 | 0 | 2.75 |
| qwen3 | color | en->zh | natural | short_answer | 12 | 0 | 12 | 7 | 7 | 0 | 1.17 |
| qwen3 | color | en->zh | natural | stop_explicit | 12 | 0 | 12 | 9 | 9 | 0 | 1.75 |
| qwen3 | color | en->zh | post_answer | plain | 12 | 0 | 11 | 8 | 8 | 0 | 2.17 |
| qwen3 | color | en->zh | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| qwen3 | color | en->zh | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 2.00 |
| qwen3 | color | zh->en | natural | plain | 12 | 0 | 12 | 4 | 2 | 0 | 1.50 |
| qwen3 | color | zh->en | natural | short_answer | 12 | 0 | 12 | 7 | 7 | 0 | 1.17 |
| qwen3 | color | zh->en | natural | stop_explicit | 12 | 0 | 12 | 9 | 9 | 0 | 1.75 |
| qwen3 | color | zh->en | post_answer | plain | 12 | 0 | 12 | 4 | 4 | 0 | 1.83 |
| qwen3 | color | zh->en | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| qwen3 | color | zh->en | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 2.00 |
| qwen3 | color | zh->zh | natural | plain | 12 | 0 | 11 | 4 | 3 | 0 | 3.50 |
| qwen3 | color | zh->zh | natural | short_answer | 12 | 0 | 12 | 9 | 9 | 0 | 1.17 |
| qwen3 | color | zh->zh | natural | stop_explicit | 12 | 0 | 12 | 10 | 10 | 0 | 1.00 |
| qwen3 | color | zh->zh | post_answer | plain | 12 | 0 | 12 | 10 | 10 | 0 | 2.17 |
| qwen3 | color | zh->zh | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.17 |
| qwen3 | color | zh->zh | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| glm4 | color | en->en | natural | plain | 12 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | color | en->en | natural | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.42 |
| glm4 | color | en->en | natural | stop_explicit | 12 | 0 | 9 | 5 | 3 | 0 | 1.33 |
| glm4 | color | en->en | post_answer | plain | 12 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | color | en->en | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 2.17 |
| glm4 | color | en->en | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| glm4 | color | zh->en | natural | plain | 12 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | color | zh->en | natural | short_answer | 12 | 0 | 12 | 11 | 11 | 0 | 1.25 |
| glm4 | color | zh->en | natural | stop_explicit | 12 | 0 | 10 | 5 | 1 | 0 | 1.83 |
| glm4 | color | zh->en | post_answer | plain | 12 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | color | zh->en | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 2.42 |
| glm4 | color | zh->en | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.08 |
| glm4 | color | zh->zh | natural | plain | 12 | 0 | 11 | 6 | 6 | 0 | 7.17 |
| glm4 | color | zh->zh | natural | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.42 |
| glm4 | color | zh->zh | natural | stop_explicit | 12 | 0 | 9 | 5 | 3 | 0 | 1.33 |
| glm4 | color | zh->zh | post_answer | plain | 12 | 0 | 9 | 5 | 5 | 0 | 7.83 |
| glm4 | color | zh->zh | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 2.17 |
| glm4 | color | zh->zh | post_answer | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.00 |
| glm4 | function | en->en | natural | plain | 9 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | function | en->en | natural | short_answer | 9 | 0 | 9 | 9 | 9 | 0 | 3.67 |
| glm4 | function | en->en | natural | stop_explicit | 9 | 0 | 9 | 6 | 6 | 0 | 3.67 |
| glm4 | function | en->en | post_answer | plain | 9 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | function | en->en | post_answer | short_answer | 9 | 0 | 7 | 4 | 4 | 0 | 3.00 |
| glm4 | function | en->en | post_answer | stop_explicit | 9 | 0 | 9 | 9 | 9 | 0 | 1.78 |
| deepseek7b | function | en->en | natural | plain | 12 | 0 | 9 | 6 | 6 | 0 | 2.33 |
| deepseek7b | function | en->en | natural | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.50 |
| deepseek7b | function | en->en | natural | stop_explicit | 12 | 0 | 12 | 12 | 12 | 0 | 1.67 |
| deepseek7b | function | en->en | post_answer | plain | 12 | 0 | 8 | 9 | 8 | 0 | 2.58 |
| deepseek7b | function | en->en | post_answer | short_answer | 12 | 0 | 12 | 12 | 12 | 0 | 1.83 |
| deepseek7b | function | en->en | post_answer | stop_explicit | 12 | 0 | 12 | 10 | 10 | 0 | 1.67 |

## Token Step Summary
| model | mode | protocol | step | stop margin | eos rank | period rank | period emitted | eos emitted | after-period continues | top tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | natural | plain | 1 | -1.90 | 92978.2 | 69.8 | 0 | 0 | 0 | {' red': 9, '红色': 6, '（': 5, ' a': 4, ' ': 4, '____': 3, ' usually': 3, '？': 2, ' brown': 2, ' black': 2, ' what': 1, ' blue': 1} |
| qwen3 | natural | plain | 2 | 2.31 | 77546.0 | 85.2 | 10 | 0 | 0 | {'.': 8, ',': 7, '，': 5, ' key': 3, '用': 3, '比如': 3, '__': 3, '（': 2, '。': 2, '?': 1, '例如': 1, ' combination': 1} |
| qwen3 | natural | plain | 3 | -3.66 | 90684.6 | 285.9 | 3 | 0 | 10 | {' but': 7, '：': 5, ' What': 4, '但': 4, ' factor': 3, '英文': 3, '。': 3, ' But': 2, 'The': 2, ' The': 2, ' If': 1, ' Also': 1} |
| qwen3 | natural | plain | 4 | -2.60 | 103920.0 | 474.5 | 0 | 0 | 13 | {' is': 4, ' some': 4, ' typical': 4, '有时': 4, ' there': 3, ' in': 3, '一个': 3, ' the': 3, ' what': 2, 'brown': 2, ' ': 2, ' a': 1} |
| qwen3 | natural | plain | 5 | -0.92 | 95005.3 | 281.1 | 0 | 0 | 13 | {' the': 4, ' color': 4, ' are': 3, ' determining': 3, ' is': 3, '词': 3, ' �': 3, '会': 2, '也': 2, ' card': 1, ' car': 1, ' apples': 1} |
| qwen3 | natural | plain | 6 | -0.61 | 94220.1 | 553.9 | 0 | 0 | 13 | {' of': 6, ' most': 3, ' have': 3, ' its': 3, '回答': 3, '是': 3, ' is': 2, ' also': 2, ' the': 2, '\n\n': 2, '可能是': 2, 'inals': 1} |
| qwen3 | natural | plain | 7 | -1.08 | 85158.5 | 328.7 | 1 | 0 | 13 | {' a': 5, ' the': 3, '：': 3, 'red': 3, '其他': 3, ' common': 2, ' value': 2, '，': 2, ' have': 1, ' red': 1, ' other': 1, ' variations': 1} |
| qwen3 | natural | plain | 8 | 1.47 | 84068.7 | 339.4 | 5 | 0 | 14 | {' color': 4, '.': 4, '颜色': 3, ' a': 2, ' colors': 2, '\n': 2, ' coat': 2, '\n\n': 2, ',': 1, ' green': 1, ' most': 1, ' for': 1} |
| qwen3 | natural | short_answer | 1 | -3.84 | 96740.0 | 43.0 | 0 | 0 | 0 | {' red': 20, '...': 10, ' black': 8, ' usually': 6, ' brown': 4} |
| qwen3 | natural | short_answer | 2 | 3.64 | 84341.0 | 22.3 | 32 | 0 | 0 | {'.': 32, '?': 10, ' dark': 4, '...': 2} |
| qwen3 | natural | short_answer | 3 | -9.45 | 54666.7 | 268.6 | 0 | 0 | 32 | {' The': 36, ' \n\n': 6, ' on': 4, '?': 2} |
| qwen3 | natural | short_answer | 4 | -0.37 | 118914.9 | 600.2 | 0 | 0 | 32 | {' color': 36, ' top': 4, 'blue': 4, ' \n\n': 2, 'The': 2} |
| qwen3 | natural | short_answer | 5 | -2.60 | 114903.3 | 68.8 | 0 | 0 | 32 | {' of': 36, ' and': 4, '\n': 4, 'Answer': 2, ' color': 2} |
| qwen3 | natural | short_answer | 6 | -8.73 | 100496.3 | 409.6 | 0 | 0 | 32 | {' a': 36, ' light': 4, '**': 4, ':': 2, ' of': 2} |
| qwen3 | natural | short_answer | 7 | 2.05 | 81430.9 | 528.6 | 0 | 0 | 32 | {' banana': 8, ' humming': 4, ' dress': 4, ' dog': 4, ' cat': 4, ' flower': 4, ' on': 4, 'blue': 4, ' bus': 2, ' steel': 2, ' a': 2, ' key': 2} |
| qwen3 | natural | short_answer | 8 | -8.31 | 91573.8 | 16.1 | 0 | 0 | 32 | {' is': 32, 'bird': 4, ' bottom': 4, '**': 4, '\n': 2, ' saw': 2} |
| qwen3 | natural | stop_explicit | 1 | -3.46 | 79147.8 | 37.2 | 0 | 0 | 0 | {' red': 18, '...': 12, ' brown': 8, ' black': 6, ' usually': 4} |
| qwen3 | natural | stop_explicit | 2 | 4.25 | 56529.5 | 24.5 | 32 | 0 | 0 | {'.': 18, '.\n': 14, '?': 8, ' dark': 4, '?\n': 4} |
| qwen3 | natural | stop_explicit | 3 | -11.83 | 38901.9 | 341.9 | 0 | 0 | 32 | {' The': 18, 'The': 10, 'Answer': 8, ' on': 4, ' \n\n': 2, ' saw': 2, ' key': 2, ' black': 2} |
| qwen3 | natural | stop_explicit | 4 | 1.33 | 87197.9 | 628.7 | 0 | 0 | 32 | {' color': 18, ' answer': 10, ':': 8, '\n': 4, ' the': 4, 'Black': 2, ' is': 2} |
| qwen3 | natural | stop_explicit | 5 | -6.81 | 81130.5 | 251.3 | 2 | 0 | 32 | {' of': 18, ' is': 10, ' red': 4, ' back': 4, ' black': 4, '.': 2, ' a': 2, 'black': 2, 'Answer': 2} |
| qwen3 | natural | stop_explicit | 6 | -3.46 | 64979.7 | 244.1 | 8 | 0 | 34 | {' a': 18, '.': 8, ' black': 6, ' brown': 4, ' and': 4, ' \n\n': 2, ' tool': 2, '\n': 2, ':': 2} |
| qwen3 | natural | stop_explicit | 7 | 4.56 | 48251.3 | 358.3 | 10 | 0 | 38 | {'.': 10, ' banana': 8, ' The': 4, ' bird': 4, 'The': 4, ' sky': 4, ' white': 4, ' Answer': 4, ' car': 2, ' used': 2, ' black': 2} |
| qwen3 | natural | stop_explicit | 8 | -6.86 | 66737.9 | 275.4 | 0 | 0 | 38 | {' is': 18, ' The': 10, ' color': 8, ' on': 4, ':': 4, ' for': 2, '\n\n': 2} |
| qwen3 | post_answer | plain | 1 | 6.59 | 61714.7 | 5.1 | 18 | 0 | 0 | {',': 16, '.': 9, '。': 9, '\n\n': 7, '\n': 4, '，': 3} |
| qwen3 | post_answer | plain | 2 | -6.82 | 71791.7 | 295.0 | 0 | 0 | 18 | {' but': 13, '正确': 7, ' ': 7, ' What': 3, '是': 3, ' But': 2, ' A': 2, ' and': 2, '对': 2, ' If': 1, ' The': 1, '但': 1} |
| qwen3 | post_answer | plain | 3 | -0.07 | 79665.4 | 414.3 | 7 | 0 | 18 | {' some': 8, '。': 7, '请': 5, ' there': 4, ' the': 4, ' is': 3, '吗': 3, '的': 2, '但是': 2, ' a': 1, ' car': 1, ' in': 1} |
| qwen3 | post_answer | plain | 4 | -1.49 | 82495.6 | 455.9 | 0 | 0 | 25 | {'用': 5, ' the': 4, ' are': 4, '，': 4, ' color': 3, '？': 3, ' card': 1, ' car': 1, ' apples': 1, ' is': 1, ' trains': 1, ' knives': 1} |
| qwen3 | post_answer | plain | 5 | -1.54 | 90209.1 | 486.1 | 0 | 0 | 25 | {'英文': 5, ' are': 4, '通常': 4, ' most': 3, ' is': 3, ' 是': 3, ' have': 2, ' also': 2, ' some': 2, '的颜色': 2, '鱼': 2, ' of': 2} |
| qwen3 | post_answer | plain | 6 | 0.56 | 82677.1 | 365.1 | 2 | 0 | 25 | {'回答': 5, ' common': 3, ' a': 3, ' the': 3, '的': 3, ' not': 2, ' of': 2, '通常': 2, ' have': 1, ' red': 1, ' other': 1, ' variations': 1} |
| qwen3 | post_answer | plain | 7 | 2.27 | 69713.6 | 345.3 | 6 | 0 | 27 | {'，': 5, ' a': 4, '。': 4, ' color': 3, '.': 2, ' of': 2, '呈': 2, ',': 1, ' green': 1, ' colors': 1, ' front': 1, ' in': 1} |
| qwen3 | post_answer | plain | 8 | -0.11 | 70950.8 | 184.0 | 3 | 0 | 29 | {' color': 4, ' for': 3, ' other': 3, ' ': 3, '.': 2, '灰色': 2, '？': 2, ' is': 2, ' in': 2, ' yellow': 1, ' then': 1, ' What': 1} |
| qwen3 | post_answer | short_answer | 1 | 7.10 | 83124.0 | 1.0 | 48 | 0 | 0 | {'.': 48} |
| qwen3 | post_answer | short_answer | 2 | -11.75 | 49151.4 | 320.5 | 0 | 0 | 48 | {' The': 48} |
| qwen3 | post_answer | short_answer | 3 | 3.24 | 137752.0 | 713.3 | 0 | 0 | 48 | {' color': 48} |
| qwen3 | post_answer | short_answer | 4 | -2.86 | 131932.7 | 75.5 | 0 | 0 | 48 | {' of': 48} |
| qwen3 | post_answer | short_answer | 5 | -10.62 | 118333.9 | 603.6 | 0 | 0 | 48 | {' a': 48} |
| qwen3 | post_answer | short_answer | 6 | 3.89 | 94115.1 | 761.3 | 0 | 0 | 48 | {' banana': 8, ' humming': 4, ' dress': 4, ' dog': 4, ' cat': 4, ' flower': 4, ' whale': 4, ' tiger': 4, ' bus': 2, ' spoon': 2, ' hammer': 2, ' key': 2} |
| qwen3 | post_answer | short_answer | 7 | -11.03 | 104828.8 | 12.4 | 0 | 0 | 48 | {' is': 44, 'bird': 4} |
| qwen3 | post_answer | short_answer | 8 | -1.85 | 79856.5 | 36.2 | 0 | 0 | 48 | {' yellow': 8, ' blue': 8, ' black': 8, ' is': 4, ' red': 4, ' pink': 4, ' orange': 4, ' silver': 2, ' also': 2, '...': 2, ' white': 2} |
| qwen3 | post_answer | stop_explicit | 1 | 8.53 | 55684.5 | 1.6 | 48 | 0 | 0 | {'.\n': 28, '.': 20} |
| qwen3 | post_answer | stop_explicit | 2 | -17.08 | 38170.2 | 427.7 | 0 | 0 | 48 | {'The': 24, ' The': 20, 'Answer': 4} |
| qwen3 | post_answer | stop_explicit | 3 | 5.05 | 103845.5 | 624.8 | 0 | 0 | 48 | {' answer': 24, ' color': 20, ':': 4} |
| qwen3 | post_answer | stop_explicit | 4 | -12.14 | 94999.1 | 213.5 | 0 | 0 | 48 | {' is': 24, ' of': 20, ' red': 4} |
| qwen3 | post_answer | stop_explicit | 5 | -3.45 | 81323.3 | 290.5 | 4 | 0 | 48 | {' a': 20, ' brown': 8, ' gray': 8, '.': 4, ' B': 4, ' GR': 4} |
| qwen3 | post_answer | stop_explicit | 6 | 9.74 | 55145.1 | 351.1 | 16 | 0 | 48 | {'.': 16, ' banana': 8, ' The': 4, ' bird': 4, ' spoon': 4, 'ROWN': 4, 'AY': 4, ' car': 2, ' key': 2} |
| qwen3 | post_answer | stop_explicit | 7 | -5.24 | 51496.5 | 224.6 | 8 | 0 | 48 | {' is': 20, ' The': 16, '.': 8, ' color': 4} |
| qwen3 | post_answer | stop_explicit | 8 | -1.35 | 72574.0 | 151.1 | 0 | 0 | 48 | {' color': 16, ' yellow': 8, ' The': 8, ' of': 4, ' blue': 4, ' silver': 4, ' red': 2, ' also': 2} |
| glm4 | natural | plain | 1 | -5.77 | 61287.7 | 26677.6 | 0 | 0 | 0 | {'!': 33, 'red': 6, '以下': 3, '\n': 1, '词': 1, '绿灯': 1} |
| glm4 | natural | plain | 2 | -5.69 | 59233.0 | 26704.6 | 0 | 0 | 0 | {'!': 33, '1': 3, '红': 2, '\n': 2, '绿': 1, '英文': 1, '是什么': 1, '片': 1, '末': 1} |
| glm4 | natural | plain | 3 | -5.50 | 61284.7 | 26723.4 | 0 | 0 | 0 | {'!': 33, '英文': 4, '颜色': 3, 'red': 1, '手机': 1, '红了': 1, '在': 1, '、': 1} |
| glm4 | natural | plain | 4 | -5.23 | 62656.2 | 26768.1 | 0 | 0 | 0 | {'!': 33, '颜色': 6, '被': 2, '是': 1, '\n': 1, '在': 1, '颜色的': 1} |
| glm4 | natural | plain | 5 | -5.14 | 63318.0 | 26680.7 | 0 | 0 | 0 | {'!': 33, '是': 6, '颜色': 2, '颜色的': 2, '：': 1, '？\n': 1} |
| glm4 | natural | plain | 6 | -5.85 | 65795.8 | 26673.7 | 0 | 0 | 0 | {'!': 33, '：': 6, '？\n': 3, '1': 1, '是': 1, '红': 1} |
| glm4 | natural | plain | 7 | -5.03 | 60720.0 | 27936.8 | 6 | 0 | 0 | {'!': 25, 'No': 8, '。\n': 6, '\n': 2, '颜色': 1, '：': 1, '\n\n': 1, '通常': 1} |
| glm4 | natural | plain | 8 | -5.57 | 60912.4 | 31754.4 | 2 | 0 | 6 | {'No': 32, '。\n': 2, '？\n': 2, '蓝': 1, '词': 1, '：': 1, '是一个': 1, '颜色': 1, '刀': 1, '尾': 1, '请注意': 1, '!': 1} |
| glm4 | natural | short_answer | 1 | -4.72 | 5620.7 | 382.3 | 0 | 0 | 0 | {' word': 36, ' this': 9} |
| glm4 | natural | short_answer | 2 | -4.56 | 5589.0 | 153.9 | 0 | 0 | 0 | {' the': 36, ' verb': 9} |
| glm4 | natural | short_answer | 3 | 0.36 | 5176.9 | 390.4 | 0 | 0 | 0 | {' banana': 32, ' is': 9, ' object': 4} |
| glm4 | natural | short_answer | 4 | -2.14 | 1181.6 | 12.1 | 0 | 0 | 0 | {' is': 29, ' to': 8, "'s": 4, 'l': 2, ' that': 1, ' as': 1} |
| glm4 | natural | short_answer | 5 | 0.14 | 2134.2 | 37.6 | 0 | 0 | 0 | {':\n': 14, ' ______': 13, ' __': 7, ' called': 2, ' transport': 2, ' determined': 1, ' ride': 1, ' be': 1, ' eat': 1, ' drive': 1, ' take': 1, ' go': 1} |
| glm4 | natural | short_answer | 6 | 0.48 | 1449.0 | 2.5 | 44 | 0 | 0 | {'.\n': 44, 'ed': 1} |
| glm4 | natural | short_answer | 7 | -0.17 | 519.5 | 8.0 | 16 | 0 | 44 | {' color': 18, '.\n': 14, ' car': 4, ' is': 4, ' .\n': 2, ' answer': 2, ' only': 1} |
| glm4 | natural | short_answer | 8 | -1.73 | 1459.1 | 6.6 | 0 | 0 | 44 | {' is': 32, ' to': 9, "'s": 2, 'jo': 2} |
| glm4 | natural | stop_explicit | 1 | -2.57 | 1759.3 | 17.0 | 0 | 0 | 0 | {' is': 23, ' to': 9, "'s": 8, ' blade': 2, 'l': 2, ' that': 1} |
| glm4 | natural | stop_explicit | 2 | 0.36 | 3652.2 | 66.4 | 0 | 0 | 0 | {':\n': 17, ' ______': 13, ' __': 7, ' blue': 2, ' transport': 2, ' ride': 1, ' be': 1, ' drive': 1, ' take': 1} |
| glm4 | natural | stop_explicit | 3 | -0.16 | 1486.2 | 20.1 | 2 | 0 | 0 | {' is': 19, ' to': 6, 'car': 4, ' ______': 2, '.\n': 2, ' __': 2, '____': 2, 'l': 2, 'h': 2, ' of': 1, ' drive': 1, ' transport': 1} |
| glm4 | natural | stop_explicit | 4 | -0.26 | 16291.3 | 163.4 | 2 | 0 | 2 | {'__': 11, 'Answer': 10, '________': 9, 'The': 7, '...\n': 2, '.\n': 2, ' people': 2, ' a': 1, ' someone': 1} |
| glm4 | natural | stop_explicit | 5 | -0.80 | 2650.7 | 180.8 | 1 | 0 | 4 | {' __': 9, ' ______': 7, ' a': 4, 'olina': 4, ' the': 3, '_': 2, ':': 2, 'Answer': 2, "'s": 2, 'ac': 2, 'azel': 2, '.\n': 1} |
| glm4 | natural | stop_explicit | 6 | -0.24 | 1062.1 | 153.4 | 9 | 0 | 4 | {':': 10, '.\n': 9, ' color': 7, '_.': 6, '__': 4, 'red': 2, 'ike': 2, ' to': 2, 'What': 1, 'The': 1, ' somewhere': 1} |
| glm4 | natural | stop_explicit | 7 | -0.10 | 2054.3 | 74.8 | 1 | 0 | 13 | {'________': 7, '\n': 7, '__': 7, ' is': 6, '_': 2, ':': 2, ' __': 2, '?\n': 2, 'Answer': 2, ' question': 2, ' to': 2, ' the': 1} |
| glm4 | natural | stop_explicit | 8 | -1.06 | 3480.9 | 119.1 | 8 | 0 | 13 | {':': 10, '.\n': 8, ' is': 6, ' word': 4, '__': 4, ' color': 3, 'Answer': 3, 'rylic': 2, ' be': 1, ' verb': 1, 'The': 1, ' somewhere': 1} |
| glm4 | post_answer | plain | 1 | -5.50 | 61353.4 | 26725.1 | 0 | 0 | 0 | {'!': 33, '英文': 4, '颜色': 3, 'red': 1, '手机': 1, '红了': 1, '在': 1, '、': 1} |
| glm4 | post_answer | plain | 2 | -5.24 | 62900.5 | 26767.9 | 0 | 0 | 0 | {'!': 33, '颜色': 6, '被': 2, '是': 1, '\n': 1, '在': 1, '颜色的': 1} |
| glm4 | post_answer | plain | 3 | -5.14 | 63763.7 | 26681.3 | 0 | 0 | 0 | {'!': 33, '是': 6, '颜色': 2, '颜色的': 2, '：': 1, '？\n': 1} |
| glm4 | post_answer | plain | 4 | -5.63 | 66378.4 | 26672.9 | 1 | 0 | 0 | {'!': 33, '：': 6, '？\n': 3, '。\n': 1, '\n': 1, '红': 1} |
| glm4 | post_answer | plain | 5 | -5.01 | 64015.8 | 26670.8 | 3 | 0 | 1 | {'!': 33, '\n': 3, '（': 3, '。\n': 2, '词': 1, '：': 1, '\n\n': 1, '。\n\n': 1} |
| glm4 | post_answer | plain | 6 | -5.31 | 60331.6 | 26669.1 | 1 | 0 | 4 | {'!': 33, '：': 3, '颜色': 2, '？\n': 2, ' red': 1, '。\n\n': 1, '\n': 1, '英文': 1, '词': 1} |
| glm4 | post_answer | plain | 7 | -5.38 | 61432.8 | 27963.1 | 0 | 0 | 5 | {'!': 25, 'No': 8, '词': 3, '\n': 3, '（': 3, 'red': 1, '\n\n': 1, '英文': 1} |
| glm4 | post_answer | plain | 8 | -6.17 | 63229.2 | 31753.9 | 0 | 0 | 5 | {'No': 32, '？\n': 3, '：': 3, ' red': 2, 'red': 1, '汽车的': 1, '红': 1, ' brown': 1, '!': 1} |
| glm4 | post_answer | short_answer | 1 | 0.34 | 5132.9 | 391.7 | 0 | 0 | 0 | {' banana': 32, ' is': 9, ' object': 4} |
| glm4 | post_answer | short_answer | 2 | -2.08 | 1181.7 | 11.5 | 0 | 0 | 0 | {' is': 29, ' to': 8, "'s": 4, 'l': 2, ' that': 1, ' as': 1} |
| glm4 | post_answer | short_answer | 3 | 0.12 | 2198.9 | 37.7 | 0 | 0 | 0 | {' ______': 19, ':\n': 14, ' called': 2, ' __': 2, ' transport': 2, ' determined': 1, ' ride': 1, ' be': 1, ' drive': 1, ' take': 1, ' go': 1} |
| glm4 | post_answer | short_answer | 4 | 2.89 | 983.8 | 2.5 | 40 | 0 | 0 | {'.\n': 36, ' people': 5, '.': 4} |
| glm4 | post_answer | short_answer | 5 | 1.03 | 1297.7 | 5.9 | 26 | 0 | 40 | {'.\n': 26, ' passengers': 5, ' is': 4, ' blue': 4, ' to': 3, 'ana': 2, ' a': 1} |
| glm4 | post_answer | short_answer | 6 | -0.43 | 1487.0 | 18.3 | 0 | 0 | 40 | {' yellow': 25, ' transport': 5, ' red': 4, ' are': 4, 'icious': 2, ' swim': 2, ' is': 1, ' carry': 1, ' a': 1} |
| glm4 | post_answer | short_answer | 7 | -0.66 | 3340.8 | 106.7 | 2 | 0 | 40 | {'_': 13, 'The': 12, '__': 6, ' is': 5, 'Answer': 2, '.\n': 2, '________': 2, ' people': 2, ' the': 1} |
| glm4 | post_answer | short_answer | 8 | -3.97 | 5519.4 | 205.2 | 0 | 0 | 40 | {'The': 19, 'Solution': 14, ' from': 4, ' color': 2, ' A': 2, ' (': 2, 'Answer': 1, ' or': 1} |
| glm4 | post_answer | stop_explicit | 1 | 3.32 | 974.0 | 2.4 | 40 | 0 | 0 | {'.\n': 39, ' people': 5, '.': 1} |
| glm4 | post_answer | stop_explicit | 2 | -7.53 | 34726.1 | 1704.1 | 4 | 0 | 40 | {'What': 33, 'Answer': 3, '.': 3, 'The': 2, 'run': 1, ' A': 1, '.\n': 1, ' from': 1} |
| glm4 | post_answer | stop_explicit | 3 | -8.72 | 9435.8 | 327.8 | 0 | 0 | 44 | {' is': 34, ' with': 3, ' A': 3, ' color': 2, ' common': 1, 'What': 1, ' one': 1} |
| glm4 | post_answer | stop_explicit | 4 | -9.14 | 7174.2 | 325.0 | 0 | 0 | 44 | {' the': 33, ' exactly': 3, ' common': 3, ' of': 2, ' use': 1, ' a': 1, ' is': 1, ' place': 1} |
| glm4 | post_answer | stop_explicit | 5 | -0.80 | 2016.7 | 586.4 | 0 | 0 | 44 | {' color': 32, ' one': 3, ' use': 3, ' a': 2, ' verb': 1, ' for': 1, ' common': 1, ' the': 1, ' to': 1} |
| glm4 | post_answer | stop_explicit | 6 | -1.12 | 694.1 | 195.8 | 1 | 0 | 44 | {' of': 32, ' English': 3, ' for': 3, ' dog': 2, '.\n': 1, ' a': 1, ' use': 1, ' verb': 1, ' another': 1} |
| glm4 | post_answer | stop_explicit | 7 | -5.73 | 3353.9 | 317.5 | 1 | 0 | 44 | {' a': 35, ' color': 2, ' is': 2, ' for': 2, 'run': 1, ' cat': 1, ' verb': 1, '.': 1} |
| glm4 | post_answer | stop_explicit | 8 | 1.07 | 2322.1 | 526.9 | 0 | 0 | 45 | {' car': 5, ' banana': 4, ' cherry': 4, ' cardinal': 4, ' train': 3, ' knife': 2, ' word': 2, ' key': 2, ' horse': 2, ' brown': 2, ' wood': 2, ' shark': 2} |
| deepseek7b | natural | plain | 1 | 1.78 | 12296.0 | 1536.4 | 0 | 0 | 0 | {' be': 5, ' drink': 1, ' cut': 1, ' carry': 1, ' fetch': 1, ' pull': 1, ' {': 1, ' blow': 1} |
| deepseek7b | natural | plain | 2 | -3.83 | 2900.2 | 326.2 | 0 | 0 | 0 | {' a': 4, ' found': 3, ' from': 1, ' it': 1, ' used': 1, ' filled': 1, 'eq': 1} |
| deepseek7b | natural | plain | 3 | 1.68 | 7087.5 | 243.8 | 0 | 0 | 0 | {' in': 3, ' to': 2, ' it': 1, ' into': 1, ' load': 1, ' ball': 1, ' cart': 1, '}\\': 1, ' bubble': 1} |
| deepseek7b | natural | plain | 4 | -0.08 | 2287.7 | 105.2 | 4 | 0 | 0 | {'.': 4, ' a': 2, ' the': 2, ' pieces': 1, ' hold': 1, ' without': 1, 'display': 1} |
| deepseek7b | natural | plain | 5 | 1.16 | 1339.4 | 163.9 | 1 | 0 | 4 | {'1': 2, ' I': 2, ' ocean': 2, '.': 1, ' something': 1, ' certain': 1, ' needing': 1, ' garden': 1, 'style': 1} |
| deepseek7b | natural | plain | 6 | 2.56 | 8637.0 | 71.4 | 1 | 0 | 5 | {',': 3, '2': 2, ' I': 1, ' point': 1, ' think': 1, "'m": 1, ' to': 1, '.': 1, ' \\': 1} |
| deepseek7b | natural | plain | 7 | -1.24 | 8637.8 | 525.9 | 0 | 0 | 6 | {' but': 2, '3': 1, ' think': 1, ' like': 1, ',': 1, ' the': 1, ' trying': 1, ' walk': 1, '6': 1, 'boxed': 1, '1': 1} |
| deepseek7b | natural | plain | 8 | -1.17 | 5305.7 | 214.3 | 0 | 0 | 6 | {' the': 2, '2': 2, '4': 1, ' a': 1, ' but': 1, ' horse': 1, ' to': 1, ',': 1, '{\\': 1, ' sometimes': 1} |
| deepseek7b | natural | short_answer | 1 | 1.57 | 7136.0 | 88.5 | 0 | 0 | 0 | {' hold': 4, ' eat': 4, ' ride': 2, ' fetch': 2} |
| deepseek7b | natural | short_answer | 2 | -0.72 | 2394.8 | 5.7 | 2 | 0 | 0 | {' fish': 4, ' something': 2, ' water': 2, '.': 2, ' objects': 2} |
| deepseek7b | natural | short_answer | 3 | 1.81 | 4338.7 | 75.7 | 10 | 0 | 2 | {'.': 10, ' So': 2} |
| deepseek7b | natural | short_answer | 4 | -7.95 | 4116.2 | 409.3 | 0 | 0 | 12 | {' So': 10, ',': 2} |
| deepseek7b | natural | short_answer | 5 | -6.72 | 7970.5 | 182.8 | 0 | 0 | 12 | {',': 10, ' the': 2} |
| deepseek7b | natural | short_answer | 6 | -6.95 | 3068.5 | 339.3 | 0 | 0 | 12 | {' the': 10, ' answer': 2} |
| deepseek7b | natural | short_answer | 7 | -1.46 | 3336.8 | 471.3 | 0 | 0 | 12 | {' answer': 6, ' verb': 4, ' is': 2} |
| deepseek7b | natural | short_answer | 8 | -8.08 | 6972.0 | 111.3 | 0 | 0 | 12 | {' is': 6, ' should': 4, ' \\': 2} |
| deepseek7b | natural | stop_explicit | 1 | 1.63 | 9013.0 | 84.8 | 0 | 0 | 0 | {' hold': 4, ' carry': 2, ' fetch': 2, ' eat': 2, ' blow': 2} |
| deepseek7b | natural | stop_explicit | 2 | -1.98 | 3347.5 | 12.3 | 0 | 0 | 0 | {' something': 2, ' water': 2, ' a': 2, ' objects': 2, ' fish': 2, ' out': 2} |
| deepseek7b | natural | stop_explicit | 3 | 3.33 | 6095.5 | 39.7 | 8 | 0 | 0 | {'.': 8, ' load': 2, ' a': 2} |
| deepseek7b | natural | stop_explicit | 4 | -3.83 | 1734.5 | 418.5 | 2 | 0 | 8 | {' So': 4, ' The': 2, ' What': 2, '.': 2, ' candle': 2} |
| deepseek7b | natural | stop_explicit | 5 | -4.59 | 3458.8 | 440.2 | 2 | 0 | 10 | {',': 4, ' verb': 2, ' is': 2, ' What': 2, '.': 2} |
| deepseek7b | natural | stop_explicit | 6 | -8.52 | 3774.2 | 485.5 | 0 | 0 | 12 | {' the': 6, ' should': 2, ' is': 2, ' What': 2} |
| deepseek7b | natural | stop_explicit | 7 | -2.78 | 3290.0 | 744.0 | 0 | 0 | 12 | {' be': 2, ' common': 2, ' the': 2, ' verb': 2, ' answer': 2, ' is': 2} |
| deepseek7b | natural | stop_explicit | 8 | -5.17 | 6749.3 | 795.8 | 0 | 0 | 12 | {' is': 4, ' in': 2, ' use': 2, ' subject': 2, ' the': 2} |
