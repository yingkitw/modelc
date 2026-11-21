//! `GET /` — the built-in single-page web UI served at the router root.
//! A static HTML page; no request logic beyond the handler wrapper.

pub(super) async fn web_ui() -> axum::response::Html<&'static str> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modelc — Local LLM Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
#header{padding:12px 20px;background:#16213e;border-bottom:1px solid #0f3460}
#header h1{font-size:18px;font-weight:600}
#header .model{font-size:12px;color:#8899aa;margin-top:2px}
#messages{flex:1;overflow-y:auto;padding:20px;max-width:900px;margin:0 auto;width:100%}
.msg{margin-bottom:16px;max-width:80%}
.msg.user{margin-left:auto}
.msg .role{font-size:11px;color:#667788;margin-bottom:4px;text-transform:uppercase}
.msg .bubble{padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.msg.user .bubble{background:#0f3460}
.msg.assistant .bubble{background:#222244;border:1px solid #333}
#input-area{padding:16px 20px;background:#16213e;border-top:1px solid #0f3460;max-width:900px;margin:0 auto;width:100%}
#input-row{display:flex;gap:8px}
#prompt{flex:1;padding:12px 16px;border:1px solid #0f3460;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:14px;resize:none;height:48px;max-height:120px}
#prompt:focus{outline:none;border-color:#533483}
#send{padding:12px 24px;background:#533483;border:none;border-radius:8px;color:#fff;font-size:14px;cursor:pointer;white-space:nowrap}
#send:hover{background:#6a4493}
#send:disabled{opacity:0.5;cursor:not-allowed}
#status{font-size:11px;color:#667788;margin-top:8px;text-align:center}
</style>
</head>
<body>
<div id="header">
<h1>modelc</h1>
<div class="model" id="model-name">Loading…</div>
</div>
<div id="messages"></div>
<div id="input-area">
<div id="input-row">
<textarea id="prompt" placeholder="Send a message… (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
<button id="send">Send</button>
</div>
<div id="status"></div>
</div>
<script>
const promptEl=document.getElementById('prompt');
const sendBtn=document.getElementById('send');
const messagesEl=document.getElementById('messages');
const statusEl=document.getElementById('status');
const modelNameEl=document.getElementById('model-name');
let busy=false;
function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  const r=document.createElement('div');
  r.className='role';r.textContent=role;
  const b=document.createElement('div');
  b.className='bubble';b.textContent=text;
  d.appendChild(r);d.appendChild(b);
  messagesEl.appendChild(d);
  messagesEl.scrollTop=messagesEl.scrollHeight;
  return b;
}
async function send(){
  if(busy)return;
  const text=promptEl.value.trim();
  if(!text)return;
  busy=true;sendBtn.disabled=true;
  promptEl.value='';promptEl.style.height='48px';
  addMsg('user',text);
  const bubble=addMsg('assistant','');
  statusEl.textContent='Generating…';
  try{
    const res=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'',messages:[{role:'user',content:text}],stream:true})
    });
    const reader=res.body.getReader();
    const decoder=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n');
      buf=lines.pop();
      for(const line of lines){
        if(line.startsWith('data: ')){
          const data=line.slice(6);
          if(data==='[DONE]')continue;
          try{
            const j=JSON.parse(data);
            const delta=j.choices?.[0]?.delta?.content||'';
            bubble.textContent+=delta;
            messagesEl.scrollTop=messagesEl.scrollHeight;
          }catch(e){}
        }
      }
    }
  }catch(e){bubble.textContent='Error: '+e.message;}
  if(!bubble.textContent)bubble.textContent='(empty response)';
  statusEl.textContent='';
  busy=false;sendBtn.disabled=false;promptEl.focus();
}
sendBtn.addEventListener('click',send);
promptEl.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
promptEl.addEventListener('input',()=>{
  promptEl.style.height='48px';
  promptEl.style.height=Math.min(promptEl.scrollHeight,120)+'px';
});
fetch('/info').then(r=>r.json()).then(j=>{
  modelNameEl.textContent=j.name+' · '+j.architecture+' · '+(j.total_params/1e6).toFixed(1)+'M params';
}).catch(()=>{modelNameEl.textContent='modelc';});
promptEl.focus();
</script>
</body>
</html>"#;
    axum::response::Html(html)
}
