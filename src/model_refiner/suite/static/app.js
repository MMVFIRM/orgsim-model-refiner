"use strict";
const el=id=>document.getElementById(id);
const show=(id,value)=>{el(id).textContent=typeof value==="string"?value:JSON.stringify(value,null,2);};
let lastPrediction=null;
el("config").value=JSON.stringify({model:"toy",optimizer_model:"toy",task:{instruction:"Route support requests.",kind:"classification",metric:"accuracy",labels:["ACCOUNT","BILLING","SHIPPING","TECH"]},max_attempts:5000},null,2);
async function api(path,method="GET",body,headers={}){
 const response=await fetch(path,{method,headers:{"Authorization":"Bearer "+el("key").value,"Content-Type":"application/json",...headers},body:body===undefined?undefined:JSON.stringify(body),cache:"no-store",credentials:"omit"});
 const result=await response.json();if(!response.ok)throw new Error(JSON.stringify(result));return result;
}
const name=()=>encodeURIComponent(el("name").value);
const version=()=>Number(el("version").value);
async function rows(id,required){const file=el(id).files[0];if(!file){if(required)throw new Error("Select a JSONL examples file.");return [];}if(file.size>4000000)throw new Error("File exceeds console size limit.");return (await file.text()).split(/\r?\n/).filter(x=>x.trim()).map(x=>JSON.parse(x));}
async function fitBody(){return {config:JSON.parse(el("config").value),records:await rows("data",true),preferences:await rows("preferences",false)};}
function bind(id,out,fn){el(id).addEventListener("click",async()=>{el(id).disabled=true;try{show(out,await fn());}catch(error){show(out,"Error: "+error.message);}finally{el(id).disabled=false;}});}
bind("connect","workspace",async()=>({info:await api("/v2/info"),registry:await api("/v2/refiners")}));
bind("plan","job",async()=>api("/v2/plan","POST",await fitBody()));
bind("fit","job",async()=>{const job=await api(`/v2/refiners/${name()}/fits`,"POST",await fitBody(),{"Idempotency-Key":crypto.randomUUID()});el("jobId").value=job.id;return job;});
bind("poll","job",async()=>api(`/v2/jobs/${encodeURIComponent(el("jobId").value)}`));
bind("cancel","job",async()=>api(`/v2/jobs/${encodeURIComponent(el("jobId").value)}`,"DELETE"));
bind("report","evaluation",async()=>{const report=await api(`/v2/refiners/${name()}/versions/${version()}/report`);return {artifact_id:report.artifact_id,gate:report.gate,arms:report.arms,comparisons:report.comparisons};});
bind("promote","evaluation",async()=>api(`/v2/refiners/${name()}/promote`,"POST",{version:version(),expected_current:el("expected").value?Number(el("expected").value):null}));
bind("predict","prediction",async()=>{lastPrediction=await api(`/v2/refiners/${name()}/predict`,"POST",{input:el("input").value,version:version()});return lastPrediction;});
bind("feedback","feedbackResult",async()=>{if(!lastPrediction)throw new Error("Make a prediction first.");const result=await api("/v2/feedback","POST",{prediction_id:lastPrediction.prediction_id,input:el("input").value,target:el("target").value});el("feedbackId").value=result.feedback_id;return result;});
for(const [id,approve] of [["approve",true],["reject",false]])bind(id,"feedbackResult",async()=>api(`/v2/feedback/${encodeURIComponent(el("feedbackId").value)}/review`,"POST",{approve}));
bind("audit","activity",async()=>api("/v2/audit"));
bind("monitor","activity",async()=>api(`/v2/refiners/${name()}/monitor`));
