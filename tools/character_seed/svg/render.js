const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
(async()=>{const [src,dst]=process.argv.slice(2);const b=await chromium.launch();const pg=await b.newPage({viewport:{width:768,height:768}});
for(const t of fs.readdirSync(src)){fs.mkdirSync(path.join(dst,t),{recursive:true});
 for(const f of fs.readdirSync(path.join(src,t))){const svg=fs.readFileSync(path.join(src,t,f),'utf8');
  await pg.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  await pg.screenshot({path:path.join(dst,t,f.replace('.svg','.png')),clip:{x:0,y:0,width:768,height:768}});}}
await b.close();})();
