# 待测试的神经元解释

每条是一个神经元（匿名编号）的职能解释。请只根据解释文字写测试文本。

- N020：表示“从旁经过/穿越”的语境中、紧接“过/past”之前的词：如 a thing of【the】past、charge【going】past、fly【b】y、疾【驰】而过
- N021：长主语（从句、疑问/“如何…”、that 从句）结束后，谓语开始处的系动词/助动词或“也/是”等词
- N022：中文论文/教案中小标题或编号（如“2.3”“学法分析”）之后、正文首字（常被空格隔开）的位置
- N023：日期区间中“to 月份”之后、日数字前的空格，或中文“年+月”中的“月”，即日期里接日数字的位置
- N024：版式异常的文本（多空格、行内折行、信息框“| 字段 =”、问答列表）中行内或字段名处的词，模式松散
- N025：百科条目开头“人名/地名 (”括号内首个 token，如生日的月份名或发音注释的“;”
- N026：数学“denoted by a bar”类记号说明句或法规条款列举中，介绍表示方式/限定条件的虚词位置
- N027：数学习题文本中“trigonometric functions/composite function/cot”等函数相关词；另含免责声明“endorsements of”
- N028：人口普查类文本中“speak…at home / schooled at home”里的“at”，以及“There were N housing units”中的 housing
- N029：“major life / body / biological”等修饰语之后的抽象名词 activities、processes、functions、events、过程
- N030：数学公式被剥离后留下空白的文本中，紧邻缺失符号的虚词或名词（and、to、of、state、space）
- N031：瑜伽、冥想、放松训练指导语中的放松动作与身体部位词，如 relax、close/open eyes、肌肉放松
- N032：讨论词语用法/语法分析的元语言文本中被引号或括号引用的词，如"in and of【itself】"、Irres【pective】
- N033：多 token 人名（多为学者、历史人物，中英文音译）的中后部子词，如 Mess【ier】、梅西【耶】、Dreyf【us】
- N035：描述距离（天体、岛屿间距离）的句子中，距离数值的开头数字或其前的空格/“约”
- N036：数学教育文本中介绍动态几何软件（Geometer's Sketchpad、GeoGebra 等）的句子，在软件名前的词或换行
- N037：法律条文/规则陈述中条件分句后的逗号或“规定，”等位置，即条件引出法律后果之处
- N038：特定词的后半 token：英文 Maced【onian】、中文“国有【控股】”“中华民族【团结】”等固定搭配的后部
- N039：代码中 for(i=0【;i】<…)、tree<…> 等紧接“<”前的标识符位置；另含若干零散中文生僻字
