import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root is missing from index.html');

// `createRoot`, not `hydrateRoot`: the markup in #root is a hand-written
// skeleton at the real layout's dimensions, not server-rendered output of
// this tree. React replaces it on first commit, and because the dimensions
// match there is no layout shift when it does.
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
