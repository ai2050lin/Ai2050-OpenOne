本文件夹用于存放微信群聊二维码图片。

文件命名规则：
- 群聊二维码：group_1.png, group_2.png, group_3.png ... （会随机显示其中一个）
- 助手二维码：assistant.png （固定显示）

请将实际的二维码图片按上述命名放入此文件夹。
join.html 中的脚本会自动检测 group_1.png 到 group_8.png 中实际存在的图片，并随机选一张显示。
