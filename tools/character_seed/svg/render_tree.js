// SVG 트리를 PNG로: node render_tree.js <svg 폴더> <png 폴더> [크기=768] [동시=4]
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
(async()=>{const [src,dst,sz='768',par='4']=process.argv.slice(2);const size=+sz;
const files=[];(function walk(d){for(const f of fs.readdirSync(d)){const p=path.join(d,f);fs.statSync(p).isDirectory()?walk(p):p.endsWith('.svg')&&files.push(p);}})(src);
const b=await chromium.launch();let i=0;
await Promise.all([...Array(+par)].map(async()=>{const pg=await b.newPage({viewport:{width:size,height:size}});
 while(i<files.length){const f=files[i++];const out=path.join(dst,path.relative(src,f)).replace(/\.svg$/,'.png');
  if(fs.existsSync(out))continue;fs.mkdirSync(path.dirname(out),{recursive:true});
  const svg=fs.readFileSync(f,'utf8').replace('<svg ',`<svg style="width:${size}px;height:${size}px" `);
  await pg.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  await pg.screenshot({path:out,clip:{x:0,y:0,width:size,height:size}});}}));
await b.close();console.log(files.length);})();
