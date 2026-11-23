/* Native-index anatomy for the offline design review. Geometry carries no activation values. */
(() => {
  const font = '"Segoe UI","Microsoft YaHei UI",sans-serif';
  const mono = 'Consolas,"SFMono-Regular",monospace';
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const blue = '#2d62d4';
  const home = { yaw: -0.52, pitch: 0.49, scale: 1 };

  window.createModelScene = function createModelScene(canvas, { onSelect, onSelectUnit, onSelectParameter, onSelectBlock, onSelectChannel } = {}) {
    const ctx = canvas.getContext('2d');
    const scene = { model: 'Qwen3-14B', layers: 40, heads: 40, width: 17408, block: 26, head: 36, unit: null, grid: true, labels: true };
    let camera = { ...home };
    let drag = null;
    let targets = [];
    let frame = 0;
    canvas.style.touchAction = 'none';
    canvas.style.cursor = 'grab';

    function requestDraw() {
      if (!frame) frame = requestAnimationFrame(draw);
    }

    function path(points, fill, stroke, width = 1) {
      ctx.beginPath();
      points.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      if (fill) {
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.lineWidth = width;
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
    }

    function text(label, x, y, { color = '#567088', size = 11, family = font, weight = 500, align = 'left' } = {}) {
      ctx.font = weight + ' ' + size + 'px ' + family;
      ctx.fillStyle = color;
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x, y);
    }

    function draw() {
      frame = 0;
      const { width, height } = canvas.getBoundingClientRect();
      if (!width || !height || !ctx) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.round(width * dpr), h = Math.round(height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const wash = ctx.createLinearGradient(0, 0, 0, height);
      wash.addColorStop(0, '#fbfdff');
      wash.addColorStop(1, '#eff5fb');
      ctx.fillStyle = wash;
      ctx.fillRect(0, 0, width, height);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      targets = [];

      const compact = width < 610;
      const scale = Math.min(width / (compact ? 465 : 850), height / 267) * camera.scale;
      const sy = Math.sin(camera.yaw), cy = Math.cos(camera.yaw);
      const sp = Math.sin(camera.pitch), cp = Math.cos(camera.pitch);
      const project = (origin, x, y, z) => [
        origin[0] + scale * (cy * x - sy * z),
        origin[1] + scale * (sp * (sy * x + cy * z) - cp * y),
      ];
      // The two stations share an orthographic camera. The right station is a magnified block.
      const stack = [width * (compact ? 0.23 : 0.325), height * 0.72];
      const detail = [width * (compact ? 0.73 : 0.718), height * 0.55];
      const stackX = compact ? 72 : 125, stackZ = compact ? 58 : 96;
      const boardX = compact ? 144 : 220, boardZ = compact ? 97 : 126;
      const { block, head, unit } = scene;
      const isHidden = scene.mode === 'hidden';
      const isParameter = scene.mode === 'parameter';
      const isGlobal = (isParameter && scene.tensorBlock == null) || (isHidden && scene.stateIndex === 0);
      const prefix = isHidden ? 'D' : 'U';
      const showUnits = unit != null && head == null;
      const showChannels = scene.channelView && scene.mode==='head';
      // A contiguous window exposes native MLP indices without sampling or inventing activations.
      const firstIndex = showUnits ? Math.floor(unit / 40) * 40 : 0;
      const visibleCount = showChannels ? scene.channels : showUnits ? Math.min(40, scene.width - firstIndex) : scene.heads;
      const stackHeight = compact ? 146 : 128;
      const layerStep = stackHeight / Math.max(1, scene.layers - 1);
      const layerY = n => n * layerStep + (n >= block ? 7 : 0) + (n > block ? 10 : 0);

      function topFace(origin, x, y, z, sx, sz) {
        return [[-1,-1], [1,-1], [1,1], [-1,1]].map(([dx,dz]) =>
          project(origin, x + dx * sx / 2, y, z + dz * sz / 2));
      }

      function box(origin, x, y, z, sx, sizeY, sz, colors, hitLayer) {
        const top = topFace(origin, x, y + sizeY, z, sx, sz);
        const base = topFace(origin, x, y, z, sx, sz);
        // Front faces follow the camera; the hidden sides stay occluded as the user rotates.
        const edges = [[0,1,-cy], [1,2,sy], [2,3,cy], [3,0,-sy]];
        for (const [a,b,facing] of edges) {
          if (facing > 0) {
            const points=[top[a], top[b], base[b], base[a]];
            path(points, a % 2 ? colors.side : colors.front, colors.line, 0.55);
            if(hitLayer!==undefined)targets.push({points,id:{layer:hitLayer}});
          }
        }
        path(top, colors.top, colors.line, 0.65);
        if(hitLayer!==undefined)targets.push({points:top,id:{layer:hitLayer}});
        return top;
      }

      function shadow(x, y, rx, ry, alpha) {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(rx, ry);
        const gradient = ctx.createRadialGradient(0, 0, 0.1, 0, 0, 1);
        gradient.addColorStop(0, 'rgba(57,88,130,' + alpha + ')');
        gradient.addColorStop(1, 'rgba(57,88,130,0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(-1,-1,2,2);
        ctx.restore();
      }

      if (scene.grid) {
        ctx.fillStyle = '#d9e4ef';
        const gap = compact ? 18 : 23;
        for (let x = gap / 2; x < width; x += gap) {
          for (let y = gap / 2; y < height; y += gap) ctx.fillRect(x, y, 1.1, 1.1);
        }
      }
      shadow(stack[0], stack[1] + 18 * scale, stackX * 0.9 * scale, 23 * scale, 0.23);
      shadow(detail[0], detail[1] + 26 * scale, boardX * 0.78 * scale, 23 * scale, 0.2);

      const selectedX = compact ? 9 : 17;
      const selectedAnchor = project(stack, selectedX + stackX / 2, layerY(block), stackZ / 2);
      const boardAnchor = project(detail, -boardX / 2 - 3, -4, -boardZ / 2);
      const elbowX = (selectedAnchor[0] + boardAnchor[0]) / 2;
      ctx.setLineDash([3,4]);
      if(!isGlobal)path([selectedAnchor, [elbowX,selectedAnchor[1]], [elbowX+10*scale,boardAnchor[1]], boardAnchor],
        null, '#88a8cc', 1);
      ctx.setLineDash([]);

      const cool = { top:'#edf3f9', side:'#b7c8db', front:'#d0deec', line:'#91a8be' };
      const selected = { top:'#8cb0fa', side:'#3c6fc7', front:'#5a88db', line:'#315fab' };
      for (let i=0; i<scene.layers; i++) {
        const active = !isGlobal && i===block;
        box(stack, active ? selectedX : 0, layerY(i), 0, stackX,
          Math.min(2,layerStep*0.58), stackZ, active ? selected : cool, onSelectBlock ? i : undefined);
      }
      // Draw the top layer's attention footprint as a small set of engraved bands.
      for (let i=1; i<=3; i++) {
        const z = (i / 4 - 0.5) * stackZ;
        const lastX = block===scene.layers-1 ? selectedX : 0;
        path([project(stack,lastX-stackX/2+6,layerY(scene.layers-1)+2,z),
          project(stack,lastX+stackX/2-6,layerY(scene.layers-1)+2,z)],null,'#cad7e7',0.8);
      }

      box(detail,0,-9,0,boardX,5,boardZ,{top:'#cedded',side:'#96abc5',front:'#b6c9de',line:'#9eb2c9'});
      box(detail,0,-4,0,boardX,4,boardZ,{top:'#eef5fd',side:'#b6cce5',front:'#d8e6f7',line:'#91aac7'});
      if (isParameter && scene.tensorShape) {
        const [heightInRows, widthInCols=1] = scene.tensorShape;
        const bands = Math.min(heightInRows, 12), columns = Math.min(widthInCols, 14);
        const cellX = (boardX-12)/columns, cellZ = (boardZ-12)/bands;
        for (let r=0;r<bands;r++) {
          const firstRow = Math.floor(r*heightInRows/bands), endRow = Math.floor((r+1)*heightInRows/bands);
          for (let c=0;c<columns;c++) {
            const firstCol = Math.floor(c*widthInCols/columns), endCol = Math.floor((c+1)*widthInCols/columns);
            const selectedCell = scene.tensorRow >= firstRow && scene.tensorRow < endRow &&
              scene.tensorCol >= firstCol && scene.tensorCol < endCol;
            const points=topFace(detail, -boardX/2+6+(c+.5)*cellX, 1, -boardZ/2+6+(r+.5)*cellZ, cellX*.96, cellZ*.94);
            path(points, selectedCell?'#3478cf':'#cadcef', '#92b2d0', .55);
            targets.push({points,id:{row:firstRow,col:firstCol}});
          }
        }
        const baseline=height-(compact?16:18);
        text(isGlobal?'模型级参数':'模型块 B'+block,stack[0],baseline,{align:'center',size:compact?10:12});
        text(scene.tensorShape.join(' × ')+' · 点击分区定位行列',detail[0],baseline,{align:'center',size:compact?10:12});
        canvas.setAttribute('aria-label',scene.model+'，'+scene.tensorLabel+'，形状 '+scene.tensorShape.join(' × ')+'，行 '+scene.tensorRow+'，列 '+scene.tensorCol);
        return;
      }
      const columns = showUnits ? Math.min(10,visibleCount) : Math.ceil(Math.sqrt(visibleCount*1.55));
      const rows = Math.ceil(visibleCount/columns);
      const gapX = (boardX-12)/columns, gapZ = (boardZ-12)/rows;
      const units = Array.from({length:visibleCount},(_,offset) => {
        const x = -boardX/2 + 6 + (offset%columns+0.5)*gapX;
        const z = -boardZ/2 + 6 + (Math.floor(offset/columns)+0.5)*gapZ;
        return { id:firstIndex+offset, x, z, depth:x*sy+z*cy };
      }).sort((a,b)=>a.depth-b.depth);
      const neutral = {top:'#dbe8f9',side:'#a4bad5',front:'#bdd0e6',line:'#95adcb'};
      const activeUnit = {top:'#4380ee',side:'#1b4d9e',front:'#2c62be',line:'#2155ae'};
      let hot;
      for (const {id,x,z} of units) {
        const active = id===(showChannels ? scene.channel : showUnits ? unit : head), topY = active ? 14 : 5;
        const points = box(detail,x,0,z,gapX*0.77,topY,gapZ*0.72,active ? activeUnit : neutral);
        targets.push({points,id});
        if (active) hot = project(detail,x,topY,z);
      }

      if (scene.labels) {
        const left = Math.max(8,stack[0]-(stackX*0.71+15)*scale);
        const yTop = project(stack,0,layerY(scene.layers-1),0)[1];
        const yBottom = stack[1];
        path([[left+4,yTop],[left,yTop],[left,yBottom],[left+4,yBottom]],null,'#9bb0c8',0.9);
        text('B'+(scene.layers-1),left-4,yTop,{align:'right',size:compact?9:11,family:mono});
        text('B0',left-4,yBottom,{align:'right',size:compact?9:11,family:mono});
        const selectedY = project(stack,0,layerY(block),0)[1];
        if (!isGlobal && selectedY>yTop+13 && selectedY<yBottom-13) {
          text('B'+block,left-4,selectedY,{align:'right',size:compact?10:12,family:mono,color:blue,weight:700});
          path([[left-1,selectedY],[left+6,selectedY]],null,blue,1.5);
        }
        if (hot) {
          const label = showChannels ? 'D'+scene.channel : showUnits ? prefix+unit : 'H'+head;
          const labelWidth = label.length*8+8;
          const tagX = clamp(detail[0]+(compact?24:64)*scale, 22, width-labelWidth-6);
          const tagY = clamp(detail[1]-67*scale, 22, height-34);
          path([hot,[hot[0]+14*scale,tagY],[tagX-3,tagY]],null,blue,1.1);
          ctx.beginPath(); ctx.arc(hot[0],hot[1],2.2,0,Math.PI*2);
          ctx.fillStyle='#fff'; ctx.fill(); ctx.lineWidth=1.4; ctx.strokeStyle=blue; ctx.stroke();
          ctx.fillStyle='#f6faff'; ctx.fillRect(tagX-3,tagY-8,labelWidth,16);
          text(label,tagX,tagY,{color:'#234e9d',family:mono,weight:700,size:compact?11:13});
        }
        const baseline = height - (compact ? 16 : 18);
        text('结构示意 · '+scene.layers+' 个块',stack[0], baseline,{align:'center',size:compact?10:12});
        const detailLabel = showChannels ? 'H'+head+' · D0–D'+(visibleCount-1) : showUnits ? (isHidden?'hs['+scene.stateIndex+']':'MLP')+' · '+prefix+firstIndex+'–'+prefix+(firstIndex+visibleCount-1) : 'B'+block+' · '+scene.heads+' 个注意力头';
        text(detailLabel,detail[0],baseline,{align:'center',size:compact?10:12});
      }
      const objectLabel = showChannels ? '/H'+head+'/D'+scene.channel+'，显示全部 '+scene.channels+' 个头内通道' : showUnits ? (isHidden?' 残差坐标 D':' 的 MLP 单元 U')+unit+'，共有 '+scene.width+' 个位置，展示 '+prefix+firstIndex+' 到 '+prefix+(firstIndex+visibleCount-1)
        : (head==null?'':' 的注意力头 H'+head);
      canvas.setAttribute('aria-label',scene.model+' 结构示意，'+(isHidden?'hs['+scene.stateIndex+']':'B'+block)+objectLabel);
    }

    function pick(x, y) {
      for (let i = targets.length - 1; i >= 0; i--) {
        const {points,id} = targets[i];
        // Convex polygon, including the edges. Search from the front for occluded tiles.
        let positive = false, negative = false;
        for (let n = 0; n < points.length; n++) {
          const [ax,ay] = points[n], [bx,by] = points[(n+1)%points.length];
          const cross = (bx-ax)*(y-ay)-(by-ay)*(x-ax);
          positive ||= cross>0.001; negative ||= cross< -0.001;
        }
        if (!(positive && negative)) return id;
      }
      return null;
    }

    canvas.addEventListener('pointerdown', event => {
      if (!event.isPrimary || event.button !== 0) return;
      drag = {x:event.clientX,y:event.clientY,id:event.pointerId,rotated:false,yaw:camera.yaw,pitch:camera.pitch};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove',event=>{
      if (drag && drag.id===event.pointerId) {
        const dx=event.clientX-drag.x, dy=event.clientY-drag.y;
        if (!drag.rotated && Math.hypot(dx,dy)<4) return;
        drag.rotated = true;
        camera.yaw = drag.yaw + dx*0.008;
        camera.pitch = clamp(drag.pitch+dy*0.006,0.22,1.12);
        canvas.style.cursor='grabbing';
        requestDraw();
      } else {
        const rect=canvas.getBoundingClientRect();
        canvas.style.cursor=pick(event.clientX-rect.left,event.clientY-rect.top)==null?'grab':'pointer';
      }
    });
    canvas.addEventListener('pointerup',event=>{
      if (!drag || event.pointerId!==drag.id) return;
      if (!drag.rotated) {
        const rect=canvas.getBoundingClientRect();
        const id=pick(event.clientX-rect.left,event.clientY-rect.top);
        if (id!=null) {
          if (id.layer!==undefined) {
            onSelectBlock?.(id.layer);
          } else if (scene.mode==='parameter') {
            onSelectParameter?.(id.row,id.col);
          } else if (scene.channelView && scene.mode==='head') {
            onSelectChannel?.(id);
          } else if (scene.unit!=null && scene.head==null) {
            scene.unit=id;
            onSelectUnit?.(scene.block,id);
          } else {
            scene.head=id;
            onSelect?.(scene.block,id);
          }
          requestDraw();
        }
      }
      drag=null;
      canvas.style.cursor='grab';
    });
    canvas.addEventListener('lostpointercapture',()=>{drag=null;canvas.style.cursor='grab';});
    function zoom(factor) {
      camera.scale = clamp(camera.scale * factor,0.65,1.75);
      requestDraw();
    }
    canvas.addEventListener('wheel',event=>{
      event.preventDefault();
      zoom(Math.exp(clamp(-event.deltaY*0.0016,-0.4,0.4)));
    },{passive:false});
    new ResizeObserver(requestDraw).observe(canvas);
    requestDraw();
    return {
      update(state) {
        Object.assign(scene,state);
        scene.layers=clamp(Math.floor(Number(scene.layers)||1),1,256);
        scene.heads=clamp(Math.floor(Number(scene.heads)||1),1,256);
        scene.width=clamp(Math.floor(Number(scene.width)||1),1,Number.MAX_SAFE_INTEGER);
        scene.block=clamp(Math.floor(Number(scene.block)||0),0,scene.layers-1);
        if (scene.head!=null) scene.head=clamp(Math.floor(Number(scene.head)||0),0,scene.heads-1);
        if (scene.unit!=null) scene.unit=clamp(Math.floor(Number(scene.unit)||0),0,scene.width-1);
        requestDraw();
      },
      reset() { camera={...home}; requestDraw(); },
      zoom,
    };
  };
})();
