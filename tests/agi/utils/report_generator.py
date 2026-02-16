"""
评估报告生成器
=============

生成HTML和JSON格式的评估报告。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..framework import EvaluationResult


class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = 'agi_test/reports'):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, result: EvaluationResult, filename: Optional[str] = None) -> str:
        """
        生成完整报告
        
        Args:
            result: 评估结果
            filename: 文件名 (可选)
        
        Returns:
            生成的报告路径
        """
        if filename is None:
            filename = f"report_{result.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 生成JSON报告
        json_path = self._generate_json(result, filename)
        
        # 生成HTML报告
        html_path = self._generate_html(result, filename)
        
        return html_path
    
    def _generate_json(self, result: EvaluationResult, filename: str) -> str:
        """生成JSON报告"""
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _generate_html(self, result: EvaluationResult, filename: str) -> str:
        """生成HTML报告"""
        filepath = self.output_dir / f"{filename}.html"
        
        html = self._build_html(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def _build_html(self, result: EvaluationResult) -> str:
        """构建HTML内容"""
        # 状态颜色
        status_color = "#4CAF50" if result.overall_passed else "#f44336"
        status_text = "PASSED" if result.overall_passed else "FAILED"
        
        # 层级HTML
        levels_html = ""
        for lv in result.levels:
            lv_color = "#4CAF50" if lv.passed else "#f44336"
            tests_html = ""
            
            for t in lv.tests:
                t_color = "#4CAF50" if t.passed else "#f44336"
                t_status = "PASS" if t.passed else "FAIL"
                tests_html += f"""
                <tr>
                    <td>{t.test_name}</td>
                    <td style="color: {t_color}">{t_status}</td>
                    <td>{t.score:.1%}</td>
                    <td>{t.threshold:.1%}</td>
                </tr>
                """
            
            levels_html += f"""
            <div class="level-card">
                <h3>Level {lv.level}: {lv.level_name}
                    <span style="color: {lv_color}; margin-left: 20px;">
                        {'PASSED' if lv.passed else 'FAILED'}
                    </span>
                </h3>
                <p>Overall Score: {lv.overall_score:.1%} | Pass Rate: {lv.pass_rate:.1%}</p>
                <table>
                    <thead>
                        <tr>
                            <th>Test</th>
                            <th>Status</th>
                            <th>Score</th>
                            <th>Threshold</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tests_html}
                    </tbody>
                </table>
            </div>
            """
        
        # 完整HTML
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGI Evaluation Report - {result.model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        .status-badge {{
            display: inline-block;
            background: {status_color};
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .summary-item .label {{
            color: #666;
        }}
        .level-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .level-card h3 {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f9f9f9;
        }}
        .progress-bar {{
            background: #eee;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AGI Evaluation Report</h1>
            <p>Model: {result.model_name}</p>
            <p>Timestamp: {result.timestamp}</p>
            <br>
            <span class="status-badge">{status_text}</span>
        </header>
        
        <div class="summary">
            <div class="summary-item">
                <div class="value">{result.total_score:.1%}</div>
                <div class="label">Total Score</div>
            </div>
            <div class="summary-item">
                <div class="value">{result.summary['levels_passed']}/5</div>
                <div class="label">Levels Passed</div>
            </div>
            <div class="summary-item">
                <div class="value">{result.summary['tests_passed']}/{result.summary['total_tests']}</div>
                <div class="label">Tests Passed</div>
            </div>
            <div class="summary-item">
                <div class="value">{result.summary['total_duration']:.1f}s</div>
                <div class="label">Duration</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {result.total_score * 100}%"></div>
        </div>
        
        {levels_html}
        
        <footer style="text-align: center; padding: 20px; color: #666;">
            <p>Generated by AGI Evaluation Framework</p>
            <p>5-Level Pyramid: Basic -> Generalize -> Agentic -> Geometric -> Safety</p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html


if __name__ == '__main__':
    print("Report Generator Module")
