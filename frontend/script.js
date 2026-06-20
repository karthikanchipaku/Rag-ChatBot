const API_URL = 'http://127.0.0.1:8000/ask';
const UPLOAD_URL = 'http://127.0.0.1:8000/upload';

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

const pdffile = document.getElementById('pdffile');
const uploadBtn = document.getElementById('uploadBtn');
const selectedFile = document.getElementById('selectedFile');

pdffile.addEventListener('change', () => {
    if (pdffile.files.length > 0) {
        selectedFile.textContent = pdffile.files[0].name;
    } else {
        selectedFile.textContent = 'No file selected';
    }
});



const uploadStatus = document.getElementById('uploadStatus');
const clearChatBtn = document.getElementById('clearChatBtn');
const toastEl = document.getElementById('toast');
const toastUndoBtn = document.getElementById('toastUndo');
const toastText = document.getElementById('toastText');
let _lastClearedSnapshot = null;
let _toastTimer = null;

let chatHistory = [];
const STORAGE_KEY = 'chat_history';
try{
  const raw = localStorage.getItem(STORAGE_KEY);
  if(raw){
    try{ chatHistory = JSON.parse(raw); }
    catch(e){
      // corrupted storage: back it up and reset
      try{ localStorage.setItem(STORAGE_KEY + '_corrupted_' + Date.now(), raw); }catch(_){ }
      localStorage.removeItem(STORAGE_KEY);
      chatHistory = [];
    }
  }else{
    chatHistory = [];
  }
}catch(e){ chatHistory = []; }

function renderHistory(){
  messagesEl.innerHTML = '';
  chatHistory.forEach(m=>{
    const d = document.createElement('div');
    d.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
    d.innerHTML = m.html;
    messagesEl.appendChild(d);
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
  // bind any source badge handlers in restored history
  bindSourceHandlers();
}

/* // Knowledge Base panel handlers
const kbToggle = document.getElementById('kbToggle');
const kbPanel = document.getElementById('kbPanel');
const kbClose = document.getElementById('kbClose');
const kbFile = document.getElementById('kbFile');
const kbUploadBtn = document.getElementById('kbUploadBtn');
const kbList = document.getElementById('kbList');

function toggleKb(open){
  if(!kbPanel) return;
  if(open === undefined) open = kbPanel.classList.contains('hidden');
  kbPanel.classList.toggle('hidden', !open);
}

if(kbToggle){ kbToggle.addEventListener('click', ()=> toggleKb(true)); }
if(kbClose){ kbClose.addEventListener('click', ()=> toggleKb(false)); }

async function refreshKbList(){
  if(!kbList) return;
  kbList.innerHTML = '<div class="small muted">Loading...</div>';
  try{
    // Best-effort: try to GET /kb/list; if unavailable, fall back to reading parse_report
    let res = await fetch('/kb/list');
    if(!res.ok) throw new Error('kb list not available');
    const js = await res.json();
    if(!Array.isArray(js)) throw new Error('invalid response');
    if(js.length===0){ kbList.innerHTML = '<div class="small muted">No indexed PDFs found</div>'; return; }
    kbList.innerHTML = js.map(item=>{
      const name = escapeHtml(item.name || item.file || 'unknown');
      const meta = escapeHtml(item.status || item.message || 'indexed');
      return `<div class="kb-item"><div><div>${name}</div><div class="meta">${meta}</div></div><div class="actions"><button class="kb-reindex" data-file="${name}">Reindex</button><button class="kb-delete" data-file="${name}">Delete</button></div></div>`;
    }).join('');
  }catch(err){
    kbList.innerHTML = '<div class="small muted">Knowledge endpoints not available on server</div>';
  }
}

if(kbUploadBtn){
  kbUploadBtn.addEventListener('click', async ()=>{
    const f = kbFile.files && kbFile.files[0];
    if(!f){ alert('Select a PDF to upload'); return; }
    kbUploadBtn.disabled = true;
    const fd = new FormData(); fd.append('file', f, f.name);
    try{
      const r = await fetch('/upload',{method:'POST',body:fd});
      const j = await r.json();
      if(r.ok && j.job_id){
        alert('Upload queued: ' + j.job_id);
        setTimeout(()=> refreshKbList(), 2000);
      }else{
        alert('Upload response: ' + (j.message || JSON.stringify(j)));
      }
    }catch(e){ alert('Upload failed: '+e.message); }
    kbUploadBtn.disabled = false;
  });
}

// delegate KB actions (reindex/delete)
if(kbList){
  kbList.addEventListener('click', async (ev)=>{
    const btn = ev.target.closest('button'); if(!btn) return;
    if(btn.classList.contains('kb-reindex')){
      const file = btn.getAttribute('data-file');
      if(!confirm('Reindex "'+file+'"?')) return;
      try{ const r = await fetch(`/kb/reindex/${encodeURIComponent(file)}`, {method:'POST'}); const j = await r.json(); alert('Reindex: '+(j.message||r.status)); refreshKbList(); }catch(e){ alert('Reindex failed: '+e.message); }
    }else if(btn.classList.contains('kb-delete')){
      const file = btn.getAttribute('data-file');
      if(!confirm('Delete "'+file+'" from KB?')) return;
      try{ const r = await fetch(`/kb/file/${encodeURIComponent(file)}`, {method:'DELETE'}); const j = await r.json(); alert('Delete: '+(j.message||r.status)); refreshKbList(); }catch(e){ alert('Delete failed: '+e.message); }
    }
  });
}

// Try to refresh KB list on load
setTimeout(()=> refreshKbList(), 800);
*/

function appendMessage(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  d.innerHTML = html;
  messagesEl.appendChild(d);
  chatHistory.push({role, html});
  try{ localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory)); }catch(e){}
  messagesEl.scrollTop = messagesEl.scrollHeight;
} 

function formatSources(sources){
  if(!Array.isArray(sources) || sources.length === 0) return '';
  const uid = 'src_' + Date.now() + '_' + Math.floor(Math.random()*1000);

  // normalize scores
  const norm = sources.map((s, idx)=> Object.assign({__idx: idx, score: (typeof s.score === 'number' ? s.score : (parseFloat(s.score) || 0))}, s));

  // group by PDF and sort chunks within each PDF by score desc
  const pdfGroups = {};
  const pdfName = s => s.metadata?.source_pdf || s.source || s.id || 'unknown';
  norm.forEach(s => {
    const name = pdfName(s);
    if(!pdfGroups[name]) pdfGroups[name] = [];
    pdfGroups[name].push(s);
  });
  Object.keys(pdfGroups).forEach(k=> pdfGroups[k].sort((a,b)=> (b.score||0)-(a.score||0)));

  // convert to array of groups with best score, sort PDFs by best score desc
  const groups = Object.keys(pdfGroups).map(name=>({name, chunks: pdfGroups[name], best: (pdfGroups[name][0] && pdfGroups[name][0].score) || 0}));
  groups.sort((a,b)=> (b.best||0)-(a.best||0));

  const pdfIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M6 2h7l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" stroke="#034ea2" stroke-width="1.2" fill="#eef6ff"/><path d="M13 2v6h6" stroke="#034ea2" stroke-width="1.2"/></svg>`;

  // render each PDF group: show only top chunk per PDF, with Show More Chunks toggle for remaining
  // show only top 3 PDFs initially to reduce noise
  const visibleCount = 3;
  const visibleGroups = groups.slice(0, visibleCount);
  const hiddenGroups = groups.slice(visibleCount);

  const renderedVisible = visibleGroups.map((g, idx)=>{
    const top = g.chunks[0];
    const bestScore = (typeof top.score === 'number') ? top.score.toFixed(3) : (top.score ?? 'n/a');
    const excerpt = top.metadata?.excerpt ? escapeHtml(String(top.metadata.excerpt).trim().replace(/\s+/g,' ').slice(0,300)) : 'Preview unavailable for this source.';
    const chunkCount = g.chunks.length;
    const header = `<div class="src-card${idx===0? ' top-source':''}"><div class="src-left">${pdfIcon}<strong>${escapeHtml(g.name)}</strong> <span class="small muted">${chunkCount} chunk${chunkCount>1?'s':''}</span></div><div class="src-right"><span class="small">⭐ ${(parseFloat(bestScore) * 100).toFixed(1)}% Match</span></div></div>`;
    const topChunkHtml = `<div class="src-entry"><div class="src-entry-header"><strong>Top chunk</strong> <span class="small">⭐ ${(parseFloat(bestScore) * 100).toFixed(1)}% Match</span></div><div class="preview">${excerpt}</div></div>`;
    let more = '';
    if(chunkCount>1){
      const restHtml = g.chunks.slice(1).map((s,i)=>{
        const score = (typeof s.score === 'number') ? s.score.toFixed(3) : (s.score ?? 'n/a');
        const preview = s.metadata?.excerpt ? escapeHtml(String(s.metadata.excerpt).trim().replace(/\s+/g,' ').slice(0,220)) : 'Preview unavailable for this source.';
        const page = s.metadata?.page || s.metadata?.page_number || s.metadata?.page_num || s.metadata?.pageIndex || null;
        const pageHtml = page ? `<div class="meta">Page: ${escapeHtml(String(page))}</div>` : '';
        return `<div class="src-entry"><div class="src-entry-header"><strong>Chunk</strong>  <span class="small">
⭐ ${(parseFloat(score) * 100).toFixed(1)}% Match
</span></div>${pageHtml}<div class="preview">${preview}</div></div>`;
      }).join('');
      more = `<div class="show-more-chunks-wrap"><button class="show-more-chunks" data-target="${uid}_pdf_${idx}">Show More (${chunkCount-1}) <span class="chev">▾</span></button><div id="${uid}_pdf_${idx}" class="src-more">${restHtml}</div></div>`;
    }
    return header + topChunkHtml + more;
  }).join('');

  let hiddenHtml = '';
  if(hiddenGroups.length > 0){
    const hiddenRendered = hiddenGroups.map((g, idx)=>{
      const top = g.chunks[0];
      const bestScore = (typeof top.score === 'number') ? top.score.toFixed(3) : (top.score ?? 'n/a');
      const excerpt = top.metadata?.excerpt ? escapeHtml(String(top.metadata.excerpt).trim().replace(/\s+/g,' ').slice(0,300)) : 'Preview unavailable for this source.';
      const chunkCount = g.chunks.length;
      const header = `<div class="src-card"><div class="src-left">${pdfIcon}<strong>${escapeHtml(g.name)}</strong> <span class="small muted">${chunkCount} chunk${chunkCount>1?'s':''}</span></div><div class="src-right">⭐ ${(parseFloat(bestScore) * 100).toFixed(0)}%</div></div>`;
      const topChunkHtml = `<div class="src-entry"><div class="src-entry-header"><strong>Top chunk</strong> <span class="small">score: ${bestScore}</span></div><div class="preview">${excerpt}</div></div>`;
      let more = '';
      if(chunkCount>1){
        const restHtml = g.chunks.slice(1).map((s,i)=>{
          const score = (typeof s.score === 'number') ? s.score.toFixed(3) : (s.score ?? 'n/a');
          const preview = s.metadata?.excerpt ? escapeHtml(String(s.metadata.excerpt).trim().replace(/\s+/g,' ').slice(0,220)) : 'Preview unavailable for this source.';
          const page = s.metadata?.page || s.metadata?.page_number || s.metadata?.page_num || s.metadata?.pageIndex || null;
          const pageHtml = page ? `<div class="meta">Page: ${escapeHtml(String(page))}</div>` : '';
          return `<div class="src-entry"><div class="src-entry-header"><strong>Chunk</strong> <span class="small">score: ${score}</span></div>${pageHtml}<div class="preview">${preview}</div></div>`;
        }).join('');
        more = `<div class="show-more-chunks-wrap"><button class="show-more-chunks" data-target="${uid}_pdf_hidden_${idx}">Show More (${chunkCount-1}) <span class="chev">▾</span></button><div id="${uid}_pdf_hidden_${idx}" class="src-more">${restHtml}</div></div>`;
      }
      return header + topChunkHtml + more;
    }).join('');

    hiddenHtml = `<div class="hidden-pdfs"><button class="show-more-pdfs" data-target="${uid}_more_pdfs">Show More PDFs (${hiddenGroups.length}) <span class="chev">▾</span></button><div id="${uid}_more_pdfs" class="src-more">${hiddenRendered}</div></div>`;
  }

  return `<div class="src-groups">${renderedVisible}</div>` + hiddenHtml;
}

function bindSourceHandlers(){
  // previous badge handlers (if any) - keep for backward compatibility
  const badges = document.querySelectorAll('.src-badge');
  badges.forEach(b=>{
    if(b._bound) return;
    b._bound = true;
    b.addEventListener('click', ()=>{
      const target = b.getAttribute('data-target');
      if(!target) return;
      const el = document.getElementById(target);
      if(!el) return;
      el.classList.toggle('visible');
      if(el.classList.contains('visible')) el.scrollIntoView({behavior:'smooth',block:'nearest'});
    });
  });

  // show-more button handler
  // show-more button handlers (for per-PDF chunks)
  const showMore = document.querySelectorAll('.show-more-chunks');
  showMore.forEach(btn=>{
    if(btn._bound) return;
    btn._bound = true;
      btn.addEventListener('click', ()=>{
      const target = btn.getAttribute('data-target');
      if(!target) return;
      const el = document.getElementById(target);
      if(!el) return;
      const isOpen = el.classList.toggle('open');
      btn.classList.toggle('open', isOpen);
      btn.innerHTML = isOpen ? `Hide <span class="chev">▾</span>` : `Show More (${el.querySelectorAll('.src-entry').length}) <span class="chev">▾</span>`;
      if(isOpen) el.scrollIntoView({behavior:'smooth', block:'nearest'});
    });
  });

  // show-more-pdfs handler
  const showPdfs = document.querySelectorAll('.show-more-pdfs');
  showPdfs.forEach(btn=>{
    if(btn._bound) return;
    btn._bound = true;
    btn.addEventListener('click', ()=>{
      const target = btn.getAttribute('data-target');
      if(!target) return;
      const el = document.getElementById(target);
      if(!el) return;
      const isOpen = el.classList.toggle('open');
      btn.classList.toggle('open', isOpen);
      btn.innerHTML = isOpen ? 'Hide' : `Show More PDFs (${el.children.length}) <span class="chev">▾</span>`;
      if(isOpen) el.scrollIntoView({behavior:'smooth', block:'nearest'});
    });
  });
}

// Clear chat handler
if(clearChatBtn){
  clearChatBtn.addEventListener('click', ()=>{
    try{
      const ok = confirm('Clear chat history for this page? This cannot be undone.');
      if(!ok) return;
      // snapshot current history for undo
      try{
        _lastClearedSnapshot = JSON.stringify(chatHistory || []);
        localStorage.setItem(STORAGE_KEY + '_last_cleared', _lastClearedSnapshot);
      }catch(_){ _lastClearedSnapshot = null; }

      // clear
      chatHistory = [];
      localStorage.removeItem(STORAGE_KEY);
      // also remove any corrupted backups
      try{
        for(const k of Object.keys(localStorage)){
          if(k.startsWith(STORAGE_KEY + '_corrupted_')) localStorage.removeItem(k);
        }
      }catch(_){ }
      messagesEl.innerHTML = '';

      // show toast with undo
      showToast('Chat cleared');
    }catch(e){ console.error('Clear chat failed', e); }
  });
}

// toast control
function showToast(msg){
  if(!toastEl) return;
  toastText.textContent = msg || 'Chat cleared';
  toastEl.classList.remove('hidden');
  // clear existing timer
  if(_toastTimer) { clearTimeout(_toastTimer); _toastTimer = null; }
  _toastTimer = setTimeout(()=>{
    hideToast();
  }, 10000);
}

function hideToast(){
  if(!toastEl) return;
  toastEl.classList.add('hidden');
  if(_toastTimer){ clearTimeout(_toastTimer); _toastTimer = null; }
  // remove persisted last_cleared snapshot when toast dismisses
  try{ localStorage.removeItem(STORAGE_KEY + '_last_cleared'); }catch(_){ }
  _lastClearedSnapshot = null;
}

if(toastUndoBtn){
  toastUndoBtn.addEventListener('click', ()=>{
    try{
      // restore snapshot from memory or storage
      let raw = _lastClearedSnapshot;
      if(!raw){ try{ raw = localStorage.getItem(STORAGE_KEY + '_last_cleared'); }catch(_){ raw = null; } }
      if(!raw) return hideToast();
      const restored = JSON.parse(raw || '[]');
      chatHistory = Array.isArray(restored) ? restored : [];
      try{ localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory)); }catch(_){ }
      renderHistory();
      hideToast();
    }catch(e){ console.error('Undo failed', e); hideToast(); }
  });
}

async function sendQuestion(q){
  if(!q || !q.trim()) return;
  appendMessage('user', `<div>${escapeHtml(q)}</div>`);
  inputEl.value = '';
  sendBtn.disabled = true;
  const askSpinner = document.getElementById('askSpinner'); askSpinner.classList.remove('hidden');
  appendMessage('assistant', '<div class="small">Thinking…</div>');

  console.debug('Sending question to API:', q);
  try{
    const res = await fetch(API_URL, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const text = await res.text();
    let data = null;
    try{ data = text ? JSON.parse(text) : null; }catch(e){ data = { raw: text }; }
    console.debug('API status:', res.status, 'body:', data);
    // remove the last assistant 'Thinking…' bubble and spinner
    const last = messagesEl.querySelector('.msg.assistant:last-child');
    if(last) last.remove();
    const askSpinner2 = document.getElementById('askSpinner'); if(askSpinner2) askSpinner2.classList.add('hidden');

    if(!res.ok){
      appendMessage('assistant', `<div class="small">Server error ${res.status}: ${escapeHtml(text)}</div>`);
      return;
    }

    const answer = data?.answer || data?.message || '(no answer)';
    // strip chunk tags like [chunk_1] if present
    const clean = answer.replace(/\[chunk_[^\]]+\]/g, '').trim();
    appendMessage('assistant', `<div>${escapeHtml(clean)}</div>` + formatSources(data?.sources));
    // bind handlers for badges in the newly added message
    bindSourceHandlers();
  }catch(err){
    console.error(err);
    // remove thinking bubble
    const last = messagesEl.querySelector('.msg.assistant:last-child');
    if(last) last.remove();
    const askSpinner2 = document.getElementById('askSpinner'); if(askSpinner2) askSpinner2.classList.add('hidden');
    appendMessage('assistant', `<div class="small">Network error: ${escapeHtml(err.message)}</div>`);
  }finally{
    sendBtn.disabled = false;
    const askSpinner2 = document.getElementById('askSpinner'); if(askSpinner2) askSpinner2.classList.add('hidden');
  }
}

function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' })[c]); }

sendBtn.addEventListener('click', ()=>{ const q = inputEl.value.trim(); if(!q) return; sendQuestion(q); });
inputEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendBtn.click(); } });

// render any saved session history
renderHistory();

uploadBtn.addEventListener('click', async ()=>{
  const f = pdffile.files && pdffile.files[0];
  if(!f){ uploadStatus.textContent = 'Select a PDF first'; return; }
  uploadStatus.textContent = 'Uploading…';
  uploadBtn.disabled = true; pdffile.disabled = true; sendBtn.disabled = true;
  const fd = new FormData(); fd.append('file', f, f.name);
  const spinner = document.getElementById('uploadSpinner'); spinner.classList.remove('hidden');
  try{
    const r = await fetch(UPLOAD_URL, {method:'POST', body: fd});
    const txt = await r.text();
    let d = null; try{ d = txt ? JSON.parse(txt) : null; }catch(e){ d = { raw: txt }; }
    console.debug('Upload response:', r.status, d);
    if(!r.ok){ uploadStatus.textContent = `Upload failed: ${txt}`; }
    else{
      const job = d?.job_id;
      if(job){
        uploadStatus.textContent = 'Queued for indexing...';
        // poll status
        let done = false;
        while(!done){
          await new Promise(r=>setTimeout(r, 2000));
          try{
            const s = await fetch(`${UPLOAD_URL}/status/${job}`);
            const sj = await s.json();
            console.debug('Job status', sj);
            uploadStatus.textContent = sj.status + (sj.message?": "+sj.message:'');
            if(sj.status === 'success'){
              appendMessage('assistant', `<div class="small">Uploaded: ${escapeHtml(f.name)} — ${escapeHtml(sj.message||'Indexed')}</div>`);
              done = true; break;
            }else if(sj.status === 'failed'){
              appendMessage('assistant', `<div class="small">Indexing failed: ${escapeHtml(sj.message||'error')}</div>`);
              done = true; break;
            }
          }catch(err){ console.error('Status poll error', err); }
        }
      }else{
        uploadStatus.textContent = d?.message || 'Upload successful';
        appendMessage('assistant', `<div class="small">Uploaded: ${escapeHtml(f.name)} — ${escapeHtml(d?.message || '')}</div>`);
      }
    }
  }catch(e){ uploadStatus.textContent = 'Upload error: '+e.message; console.error(e); }
  spinner.classList.add('hidden'); uploadBtn.disabled = false; pdffile.disabled = false; sendBtn.disabled = false;
  setTimeout(()=>{ uploadStatus.textContent = ''; }, 4000);
});
const placeholderTexts = [
"Ask about your PDF...",
"Search documents...",
"Ask AI anything..."
];

let current = 0;

setInterval(()=>{

inputEl.placeholder =
placeholderTexts[current];

current =
(current + 1) %
placeholderTexts.length;

},2000);
