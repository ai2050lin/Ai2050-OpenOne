
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Activity, Network, Play, Zap } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

const FiberNetPanel = () => {
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('I love her');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  const presets = {
    en: "I love her",
    fr: "Je aime la"
  };

  useEffect(() => {
    setInputText(presets[language] || "");
    setResult(null);
  }, [language]);

  const handleInference = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/fibernet/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, lang: language })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Inference failed:", error);
    } finally {
      setLoading(false);
    }
  };

  // Draw Heatmap
  useEffect(() => {
    if (!result || !result.attention || !canvasRef.current) return;
    
    // We visualize Layer 0 Attention (Logic Structure)
    const attn = result.attention[0]; // [Seq, Seq]
    const tokens = result.tokens;
    const size = attn.length;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const cellSize = 50;
    const padding = 40;
    
    canvas.width = size * cellSize + padding * 2;
    canvas.height = size * cellSize + padding * 2;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw cells
    for (let i = 0; i < size; i++) { // Query (Rows)
      for (let j = 0; j < size; j++) { // Key (Cols)
        const weight = attn[i][j];
        // Color mapping: White to Blue
        const intensity = Math.floor((1 - weight) * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`;
        ctx.fillRect(padding + j * cellSize, padding + i * cellSize, cellSize, cellSize);
        
        // Text
        ctx.fillStyle = weight > 0.5 ? 'white' : 'black';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(weight.toFixed(2), padding + j * cellSize + cellSize/2, padding + i * cellSize + cellSize/2);
      }
    }
    
    // Draw Labels
    ctx.fillStyle = 'black';
    ctx.font = '14px sans-serif';
    
    // X-axis (Keys)
    ctx.textAlign = 'center';
    tokens.forEach((token, idx) => {
        ctx.fillText(token, padding + idx * cellSize + cellSize/2, padding - 10);
    });
    
    // Y-axis (Queries)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    tokens.forEach((token, idx) => {
        ctx.fillText(token, padding - 10, padding + idx * cellSize + cellSize/2);
    });
    
    // Title
    ctx.textAlign = 'center';
    ctx.fillText("Logic Stream Attention Pattern", canvas.width/2, canvas.height - 10);
    
  }, [result]);

  return (
    <div className="h-full flex flex-col p-4 bg-slate-50 overflow-y-auto">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-6 w-6 text-indigo-600" />
            FiberNet Interactive Lab
            <Badge variant="outline" className="ml-2">Phase 14</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 mb-4">
             <div className="flex bg-slate-200 rounded p-1">
                <Button 
                    variant={language === 'en' ? 'default' : 'ghost'} 
                    onClick={() => setLanguage('en')}
                    size="sm"
                >
                    English (Logic Source)
                </Button>
                <Button 
                    variant={language === 'fr' ? 'default' : 'ghost'} 
                    onClick={() => setLanguage('fr')}
                    size="sm"
                >
                    French (Frozen Transfer)
                </Button>
             </div>
             <Input 
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="flex-1"
             />
             <Button onClick={handleInference} disabled={loading}>
                {loading ? <Activity className="animate-spin mr-2" /> : <Play className="mr-2" />}
                Run Inference
             </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
                <CardHeader><CardTitle className="text-sm">Logic Stream Visualization</CardTitle></CardHeader>
                <CardContent className="flex justify-center bg-white p-4 rounded border">
                    {result ? (
                        <canvas ref={canvasRef} />
                    ) : (
                        <div className="text-slate-400 h-64 flex items-center">Run inference to see logic topology</div>
                    )}
                </CardContent>
            </Card>
            
            <Card>
                <CardHeader><CardTitle className="text-sm">Analysis Result</CardTitle></CardHeader>
                <CardContent>
                    {result ? (
                        <div className="space-y-4">
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-bold">Topology</div>
                                <div className="text-lg font-mono text-indigo-700">Sequence Length: {result.tokens.length}</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-bold">Prediction</div>
                                <div className="text-2xl font-bold flex items-center gap-2">
                                    {result.tokens.join(" ")} <Zap className="h-4 w-4 text-yellow-500" /> {result.next_token}
                                </div>
                            </div>
                            <div className="bg-slate-100 p-3 rounded text-sm text-slate-600">
                                <strong>Insight:</strong> Observe how the Attention Heatmap on the left remains structurally similar (e.g., Verb attending to Subject) regardless of whether you use English or French. This confirms <strong>Logic Decoupling</strong>.
                            </div>
                        </div>
                    ) : (
                         <div className="text-slate-400 h-64 flex items-center justify-center">Waiting for input...</div>
                    )}
                </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default FiberNetPanel;
