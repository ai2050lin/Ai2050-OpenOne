import { forwardRef, useCallback, useEffect, useRef } from 'react';
import { OrbitControls } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { MOUSE } from 'three';

const BUTTONS = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE };
const MODIFIED_BUTTONS = { ...BUTTONS, LEFT: MOUSE.ROTATE, RIGHT: MOUSE.PAN };

export default forwardRef(function SceneOrbitControls({ domElement, ...props }, ref) {
  const target = useThree(state => domElement || state.events.connected || state.gl.domElement);
  const controls = useRef(null);
  const setRef = useCallback(value => {
    controls.current = value;
    if (typeof ref === 'function') ref(value);
    else if (ref) ref.current = value;
  }, [ref]);
  useEffect(() => {
    const setButtons = event => {
      // OrbitControls swaps rotate and pan when any of these modifiers are held.
      if (controls.current) controls.current.mouseButtons = event.ctrlKey || event.metaKey || event.shiftKey ? MODIFIED_BUTTONS : BUTTONS;
    };
    target.addEventListener('pointerdown', setButtons, true);
    return () => target.removeEventListener('pointerdown', setButtons, true);
  }, [target]);
  return <OrbitControls {...props} ref={setRef} domElement={target} mouseButtons={BUTTONS} />;
});
