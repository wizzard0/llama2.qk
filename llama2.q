/ Llama2 inference in Q (kdb+ 4.0)
/ Usage: q llama2.q -checkpoint stories15M.bin -tokenizer tokenizer.bin -prompt "Once upon a time"
/ 76 SLOC

/ RNG (XorShift64 for llama2.c compat)
seed:0j
bxor:{0b sv(0b vs x)<>0b vs y}
band:{0b sv(0b vs x)and 0b vs y}
ushr:{[x;n]b:0b vs x;0b sv(n#0b),neg[n]_b}
xsh:{s::seed;s::bxor[s;ushr[s;12]];s::bxor[s;band[s*33554432;-1j]];s::bxor[s;ushr[s;27]];seed::s;s}
ru32:{s:xsh[];(ushr[s*2685821657736338717j;32])mod 4294967296}
rf32:{first"e"$enlist(ru32[]div 256)%16777216.0}

/ Binary reading via IPC deserialize (-9!)
/ Header: 0x01000000 (LE) + len (4b) + type (1b) + attr (1b) + count (4b) + data
le4:{reverse -4#0x0 vs x}                                       / int to 4 LE bytes
ipc:{[t;n;d] -9!0x01000000,(le4 14+count d),t,0x00,(le4 n),d}   / build IPC msg and parse
ri:{[p;o;n]ipc[0x06;n;read1(p;o;4*n)]}                          / type 0x06 = int list
rf:{[p;o;n]ipc[0x08;n;read1(p;o;4*n)]}

/ Config
rdcfg:{[p]h:ri[p;0;7];v:abs h 5;
 `dim`hdim`nl`nh`nkv`vsz`sl`sh`hs!(h 0;h 1;h 2;h 3;h 4;v;h 6;h[5]>0;h[0]div h 3)}

/ Weights
rdw:{[p;c]o:28;d:c`dim;nl:c`nl;hd:c`hdim;vz:c`vsz;sl:c`sl;hs:c`hs;w:()!();
 n:vz*d;w[`emb]:rf[p;o;n];o+:4*n;w[`rmsA]:(nl;d)#rf[p;o;nl*d];o+:4*nl*d;
 n:nl*d*d;w[`wq]:`float$(nl;d;d)#rf[p;o;n];o+:4*n;w[`wk]:`float$(nl;d;d)#rf[p;o;n];o+:4*n;
 w[`wv]:`float$(nl;d;d)#rf[p;o;n];o+:4*n;w[`wo]:`float$(nl;d;d)#rf[p;o;n];o+:4*n;
 w[`rmsF]:(nl;d)#rf[p;o;nl*d];o+:4*nl*d;
 n:nl*hd*d;w[`w1]:`float$(nl;hd;d)#rf[p;o;n];o+:4*n;
 n:nl*d*hd;w[`w2]:`float$(nl;d;hd)#rf[p;o;n];o+:4*n;
 n:nl*hd*d;w[`w3]:`float$(nl;hd;d)#rf[p;o;n];o+:4*n;
 w[`rmsO]:rf[p;o;d];o+:4*d;n:sl*hs div 2;w[`fcr]:rf[p;o;n];o+:4*n;w[`fci]:rf[p;o;n];o+:4*n;
 wcls:$[c`sh;w`emb;rf[p;o;vz*d]];w[`wcls]:`float$(vz;d)#wcls;w}

/ Tokenizer
rdtok:{[p;n]o:4;v:n#enlist"";sc:n#0e;i:0;
 while[i<n;sc[i]:first rf[p;o;1];ln:first ri[p;o+4;1];v[i]:`char$read1(p;o+8;ln);o+:8+ln;i+:1];(v;sc)}

/ BPE encode
bpestep:{[v;sc;tk]bs:-1e38;bi:-1;bx:-1;i:0;
 while[i<-1+count tk;mg:v[tk i],v[tk i+1];id:v?mg;if[(id<count v)&sc[id]>bs;bs:sc id;bi:id;bx:i];i+:1];
 $[bx<0;tk;((bx#tk),bi),(bx+2)_tk]}
bpe:{[t;v;sc]tk:v?/:enlist each t;if[any tk=count v;'"char not in vocab"];bpestep[v;sc]/[tk]}

/ Core ops
rms:{[x;w]r:1%sqrt 1e-5+(sum x*x)%count x;`real$w*r*x}
mmf:{[w;x]`real$w mmu`float$x}                         / 2x memory bandwidth!
sm:{[x]e:exp x-max x;`real$e%sum e}
silu:{[x]`real$x%1+exp neg x}

/ State
inist:{[c]`x`xb`xb2`hb`hb2`q`k`v`att`lg`kc`vc!(c[`dim]#0e;c[`dim]#0e;c[`dim]#0e;c[`hdim]#0e;c[`hdim]#0e;
 c[`dim]#0e;c[`dim]#0e;c[`dim]#0e;(c[`nh]*c[`sl])#0e;c[`vsz]#0e;
 (c[`nl]*c[`sl]*c[`dim])#0e;(c[`nl]*c[`sl]*c[`dim])#0e)}

/ RoPE
ropeV:{[w;pos;hs;d;q;k]idx:2*til d div 2;fi:(pos*hs div 2)+(idx mod hs)div 2;fcr:w[`fcr]fi;fci:w[`fci]fi;
 q0:q idx;q1:q idx+1;k0:k idx;k1:k idx+1;
 (`real$raze((q0*fcr)-q1*fci),'(q0*fci)+q1*fcr;`real$raze((k0*fcr)-k1*fci),'(k0*fci)+k1*fcr)}

/ Attention for one head
attn:{[s;loff;pos;hs;sl;d;h]qh:hs#(h*hs)_s`q;ho:h*hs;np:1+pos;
 idx:raze((til np)*d)+\:loff+ho+til hs;
 ky:hs cut(s`kc)idx;sc:`real$(sum each ky*\:qh)%sqrt hs;
 e:exp sc-max sc;at:`real$e%sum e;
 vl:hs cut(s`vc)idx;s[`xb;ho+til hs]:`real$sum at*vl;s}

/ One layer
layer:{[pos;c;w;s;l]d:c`dim;hd:c`hdim;hs:c`hs;nh:c`nh;sl:c`sl;
 s[`xb]:rms[s`x;w[`rmsA]l];s[`q]:mmf[w[`wq;l];s`xb];s[`k]:mmf[w[`wk;l];s`xb];s[`v]:mmf[w[`wv;l];s`xb];
 qk:ropeV[w;pos;hs;d;s`q;s`k];s[`q]:qk 0;s[`k]:qk 1;
 loff:l*sl*d;kvo:loff+pos*d;s[`kc;kvo+til d]:s`k;s[`vc;kvo+til d]:s`v;
 s:attn[;loff;pos;hs;sl;d]/[s;til nh];s[`xb2]:mmf[w[`wo;l];s`xb];s[`x]+:s`xb2;
 s[`xb]:rms[s`x;w[`rmsF]l];s[`hb]:mmf[w[`w1;l];s`xb];s[`hb2]:mmf[w[`w3;l];s`xb];
 s[`hb]:silu[s`hb]*s`hb2;s[`xb]:mmf[w[`w2;l];s`hb];s[`x]+:s`xb;s}

/ Forward
fwd:{[tok;pos;c;s;w]d:c`dim;s[`x]:d#(tok*d)_w`emb;s:layer[pos;c;w]/[s;til c`nl];
 s[`x]:rms[s`x;w`rmsO];s[`lg]:mmf[w`wcls;s`x];s}

/ Sampling
amax:{first where x=max x}
samp:{[lg;vz]rv:rf32[]*sum lg;i:0;cp:0e;while[i<vz;cp+:lg i;if[rv<cp;:i];i+:1];0}
sampp:{[lg;tp]ix:idesc lg;pr:lg ix;cp:sums pr;co:first where cp>tp;if[null co;co:count pr];co+:1;
 pr:co#pr;ix:co#ix;rv:rf32[]*sum pr;i:0;cp:0e;while[i<co;cp+:pr i;if[rv<cp;:ix i];i+:1];first ix}
sampn:{[lg;tmp;tp;vz]if[tmp=0e;:amax lg];lg:sm lg%tmp;$[(tp<1)&tp>0;sampp[lg;tp];samp[lg;vz]]}

/ Generation
gen:{[ckpt;tokp;steps;tmp;tp;sd;prompt]
 -1"Loading...";cfg:rdcfg ckpt;
 -1"dim=",string[cfg`dim]," layers=",string[cfg`nl]," heads=",string[cfg`nh]," vocab=",string cfg`vsz;
 wts:rdw[ckpt;cfg];-1"Loaded weights";tok:rdtok[tokp;cfg`vsz];-1"Tokenizer: ",string count tok 0;
 seed::$[sd=0;`long$.z.t;sd];if[(steps<1)|steps>cfg`sl;steps:cfg`sl];
 ptok:$[0<count prompt;bpe[prompt;tok 0;tok 1];`int$()];-1"Prompt tokens: ",string count ptok;-1"";
 st:inist cfg;tk:1;pos:0;t0:0;out:"";
 while[pos<steps;st:fwd[tk;pos;cfg;st;wts];nxt:$[pos<count ptok;ptok pos;sampn[st`lg;tmp;tp;cfg`vsz]];pos+:1;
  if[nxt=1;-1"";-1"tok/s: ",string(pos-1)%(`long$.z.p-t0)%1e9;:out];
  ts:tok[0]nxt;if[(tk=1)&0<count ts;if[ts[0]=" ";ts:1_ts]];1 ts;out,:ts;tk:nxt;if[t0=0;t0:.z.p]];
 -1"";-1"tok/s: ",string(pos-1)%(`long$.z.p-t0)%1e9;out}

/ CLI
usage:{-2"Usage: q llama2.q -checkpoint <model.bin> [opts]";-2"-tokenizer -temp -topp -seed -steps -prompt";exit 1}
main:{a:.Q.opt .z.x;if[not`checkpoint in key a;usage[]];ck:first a`checkpoint;if[0=count ck;usage[]];
 tp:$[`tokenizer in key a;first a`tokenizer;"tokenizer.bin"];
 tmp:$[`temp in key a;"F"$first a`temp;1.0];topp:$[`topp in key a;"F"$first a`topp;0.9];
 sd:$[`seed in key a;"J"$first a`seed;0j];st:$[`steps in key a;"I"$first a`steps;256i];
 pr:$[`prompt in key a;first a`prompt;""];gen[hsym`$ck;hsym`$tp;st;tmp;topp;sd;pr];exit 0}
if[count .z.x;main[]]
