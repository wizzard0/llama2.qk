/ Llama2 inference in Q (kdb+ 4.0)
/ Usage: q llama2.q -checkpoint stories15M.bin -tokenizer tokenizer.bin -prompt "Once upon a time"
/ 69 SLOC

/ RNG (XorShift64 for llama2.c compat)
seed:0j
bxor:{0b sv(0b vs x)<>0b vs y}
band:{0b sv(0b vs x)and 0b vs y}
ushr:{[x;n]0b sv(n#0b),neg[n]_0b vs x}
xsh:{seed::bxor[seed;ushr[seed;12]];seed::bxor[seed;band[seed*33554432;-1j]];seed::bxor[seed;ushr[seed;27]];seed}
ru32:{(ushr[xsh[]*2685821657736338717j;32])mod 4294967296}
rf32:{first"e"$enlist(ru32[]div 256)%16777216.0}

/ Binary reading via IPC deserialize (-9!)
/ Header: 0x01000000 (LE) + len (4b) + type (1b) + attr (1b) + count (4b) + data
le4:{reverse -4#0x0 vs x}                                       / int to 4 LE bytes
ipc:{[t;n;d] -9!0x01000000,(le4 14+count d),t,0x00,(le4 n),d}   / build IPC msg and parse
ri:{[p;o;n]ipc[0x06;n;read1(p;o;4*n)]}                          / type 0x06 = int list
rf:{[p;o;n]ipc[0x08;n;read1(p;o;4*n)]}

/ Config
rdcfg:{[p]h:ri[p;0;7];
 dim::h 0;hdim::h 1;nl::h 2;nh::h 3;nkv::h 4;vsz::abs h 5;sl::h 6;shared::h[5]>0;hs::dim div nh}

/ Weights
rdw:{[p]o:28;n:vsz*dim;emb::rf[p;o;n];o+:4*n;rmsA::(nl;dim)#rf[p;o;nl*dim];o+:4*nl*dim;
 n:nl*dim*dim;wq::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;wk::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;
 wv::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;wo::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;
 rmsF::(nl;dim)#rf[p;o;nl*dim];o+:4*nl*dim;
 n:nl*hdim*dim;w1::`float$(nl;hdim;dim)#rf[p;o;n];o+:4*n;
 n:nl*dim*hdim;w2::`float$(nl;dim;hdim)#rf[p;o;n];o+:4*n;
 n:nl*hdim*dim;w3::`float$(nl;hdim;dim)#rf[p;o;n];o+:4*n;
 rmsO::rf[p;o;dim];o+:4*dim;n:sl*hs div 2;fcr::rf[p;o;n];o+:4*n;fci::rf[p;o;n];o+:4*n;
 wcls::`float$(vsz;dim)#$[shared;emb;rf[p;o;vsz*dim]]}

/ Tokenizer
rdtok:{[p;n]o:4;vocab::n#enlist"";scores::n#0e;i:0;
 while[i<n;scores[i]::first rf[p;o;1];ln:first ri[p;o+4;1];vocab[i]::`char$read1(p;o+8;ln);o+:8+ln;i+:1]}

/ BPE encode
bpestep:{[tk]bs:-1e38;bi:-1;bx:-1;i:0;
 while[i<-1+count tk;mg:vocab[tk i],vocab[tk i+1];id:vocab?mg;if[(id<count vocab)&scores[id]>bs;bs:scores id;bi:id;bx:i];i+:1];
 $[bx<0;tk;((bx#tk),bi),(bx+2)_tk]}
bpe:{[t]tk:vocab?/:enlist each t;if[any tk=count vocab;'"char not in vocab"];bpestep/[tk]}

/ Core ops
rms:{[w;x]r:1%sqrt 1e-5+(sum x*x)%count x;`real$w*r*x}
mmf:{[w;x]`real$w mmu`float$x}                         / 2x memory bandwidth!
softmax:{[x]e:exp x-max x;`real$e%sum e}
silu:{[x]`real$x%1+exp neg x}

/ State
inist:{x::dim#0e;xb::dim#0e;xb2::dim#0e;hb::hdim#0e;hb2::hdim#0e;
 q::dim#0e;k::dim#0e;v::dim#0e;att::(nh*sl)#0e;lg::vsz#0e;
 kc::(nl*sl*dim)#0e;vc::(nl*sl*dim)#0e}

/ RoPE
rope:{[pos]idx:2*til dim div 2;fi:(pos*hs div 2)+(idx mod hs)div 2;
 fc:fcr fi;fs:fci fi;q0:q idx;q1:q idx+1;k0:k idx;k1:k idx+1;
 q::`real$raze((q0*fc)-q1*fs),'(q0*fs)+q1*fc;k::`real$raze((k0*fc)-k1*fs),'(k0*fs)+k1*fc}

/ Attention for one head
attn:{[loff;pos;h]qh:hs#(h*hs)_q;ho:h*hs;np:1+pos;
 idx:raze((til np)*dim)+\:loff+ho+til hs;ky:hs cut kc idx;sc:`real$(sum each ky*\:qh)%sqrt hs;
 e:exp sc-max sc;at:`real$e%sum e;vl:hs cut vc idx;xb[ho+til hs]::`real$sum at*vl}

/ One layer
layer:{[pos;l]xb::rms[rmsA l;x];q::mmf[wq l;xb];k::mmf[wk l;xb];v::mmf[wv l;xb];
 rope pos;loff:l*sl*dim;kvo:loff+pos*dim;kc[kvo+til dim]::k;vc[kvo+til dim]::v;
 attn[loff;pos]each til nh;xb2::mmf[wo l;xb];x::x+xb2;
 xb::rms[rmsF l;x];hb::mmf[w1 l;xb];hb2::mmf[w3 l;xb];
 hb::silu[hb]*hb2;xb::mmf[w2 l;hb];x::x+xb}

/ Forward
fwd:{[tok;pos]x::dim#(tok*dim)_emb;layer[pos]each til nl;x::rms[rmsO;x];lg::mmf[wcls;x]}

/ Sampling
amax:{first where x=max x}
samp:{[lg]rv:rf32[]*sum lg;i:0;cp:0e;while[i<vsz;cp+:lg i;if[rv<cp;:i];i+:1];0}
sampp:{[lg;tp]ix:idesc lg;pr:lg ix;cp:sums pr;co:first where cp>tp;if[null co;co:count pr];co+:1;
 pr:co#pr;ix:co#ix;rv:rf32[]*sum pr;i:0;cp:0e;while[i<co;cp+:pr i;if[rv<cp;:ix i];i+:1];first ix}
sampn:{[tmp;tp]if[tmp=0e;:amax lg];l:softmax lg%tmp;$[(tp<1)&tp>0;sampp[l;tp];samp l]}

/ Generation
gen:{[ckpt;tokp;steps;tmp;topp;sd;prompt]
 -1"Loading...";rdcfg ckpt;-1"dim=",string[dim]," layers=",string[nl]," heads=",string[nh]," vocab=",string vsz;
 rdw ckpt;-1"Loaded weights";rdtok[tokp;vsz];-1"Tokenizer: ",string count vocab;
 seed::$[sd=0;`long$.z.t;sd];if[(steps<1)|steps>sl;steps:sl];
 ptok:$[0<count prompt;bpe prompt;`int$()];-1"Prompt tokens: ",string count ptok;-1"";
 inist[];tk:1;pos:0;t0:0;out:"";
 while[pos<steps;fwd[tk;pos];nxt:$[pos<count ptok;ptok pos;sampn[tmp;topp]];pos+:1;
  if[nxt=1;-1"";-1"tok/s: ",string(pos-1)%(`long$.z.p-t0)%1e9;:out];
  ts:vocab nxt;if[(tk=1)&0<count ts;if[ts[0]=" ";ts:1_ts]];1 ts;out,:ts;tk:nxt;if[t0=0;t0:.z.p]];
 -1"";-1"tok/s: ",string(pos-1)%(`long$.z.p-t0)%1e9;out}

/ CLI
usage:{-2"Usage: q llama2.q -checkpoint <model.bin> [opts]";-2"-tokenizer -temp -topp -seed -steps -prompt";exit 1}
main:{a:.Q.opt .z.x;if[not`checkpoint in key a;usage[]];ck:first a`checkpoint;if[0=count ck;usage[]];
 tp:$[`tokenizer in key a;first a`tokenizer;"tokenizer.bin"];
 tmp:$[`temp in key a;"F"$first a`temp;1.0];topp:$[`topp in key a;"F"$first a`topp;0.9];
 sd:$[`seed in key a;"J"$first a`seed;0j];st:$[`steps in key a;"I"$first a`steps;256i];
 pr:$[`prompt in key a;first a`prompt;""];gen[hsym`$ck;hsym`$tp;st;tmp;topp;sd;pr];exit 0}
if[count .z.x;main[]]
