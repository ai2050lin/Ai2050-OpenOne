import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import AnnotationApp from './AnnotationApp.jsx'
import './annotation.css'

createRoot(document.getElementById('annotation-root')).render(
  <StrictMode>
    <AnnotationApp />
  </StrictMode>,
)
