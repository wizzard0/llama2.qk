/ Llama2 inference in pure K (kdb+ 4.0)
/ Usage: q llama2.k -checkpoint stories15M.bin -tokenizer tokenizer.bin -prompt "Once upon a time"
/ 71 SLOC

/ Helpers
md:{x-y*x div y}
cut:{[n;x]((#x)div n;n)#x}

/ RNG (XorShift64 for llama2.c compat)
seed:0j
bxor:{0b/:~(0b\:x)=0b\:y}
band:{0b/:(0b\:x)&0b\:y}
ushr:{[x;n]0b/:(n#0b),(-n)_0b\:x}
xsh:{seed::bxor[seed;ushr[seed;12]];seed::bxor[seed;band[seed*33554432;-1j]];seed::bxor[seed;ushr[seed;27]];seed}
ru32:{md[ushr[xsh[]*2685821657736338717j;32];4294967296]}
rf32:{*"e"$,(ru32[]div 256)%16777216.0}

/ Binary reading via IPC deserialize (-9!)
/ Header: 0x01000000 (LE) + len (4b) + type (1b) + attr (1b) + count (4b) + data
le4:{|(-4)#0x0\:x}                                             / int to 4 LE bytes
ipc:{[t;n;d] -9!0x01000000,(le4 14+#d),t,0x00,(le4 n),d}       / build IPC msg and parse
ri:{[p;o;n]`long$ipc[0x06;n;1:(p;o;4*n)]}                      / type 0x06 = int list
rf:{[p;o;n]ipc[0x08;n;1:(p;o;4*n)]}

/ Config
rdcfg:{[p]h:ri[p;0;7];
 dim::h 0;hdim::h 1;nl::h 2;nh::h 3;nkv::h 4;vsz::$[h[5]<0;-h 5;h 5];sl::h 6;shared::h[5]>0;hs::dim div nh}

/ Weights
rdw:{[p]o:28;n:vsz*dim;emb::rf[p;o;n];o+:4*n;rmsA::(nl;dim)#rf[p;o;nl*dim];o+:4*nl*dim
 n:nl*dim*dim;wq::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;wk::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n
 wv::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n;wo::`float$(nl;dim;dim)#rf[p;o;n];o+:4*n
 rmsF::(nl;dim)#rf[p;o;nl*dim];o+:4*nl*dim
 n:nl*hdim*dim;w1::`float$(nl;hdim;dim)#rf[p;o;n];o+:4*n
 n:nl*dim*hdim;w2::`float$(nl;dim;hdim)#rf[p;o;n];o+:4*n
 n:nl*hdim*dim;w3::`float$(nl;hdim;dim)#rf[p;o;n];o+:4*n
 rmsO::rf[p;o;dim];o+:4*dim;n:sl*hs div 2;fcr::rf[p;o;n];o+:4*n;fci::rf[p;o;n];o+:4*n
 wcls::`float$(vsz;dim)#$[shared;emb;rf[p;o;vsz*dim]]}

/ Tokenizer
rdtok:{[p;n]o:4;vocab::n#,"";scores::n#0e;i:0
 while[i<n;scores[i]::*rf[p;o;1];ln:*ri[p;o+4;1];vocab[i]::`char$1:(p;o+8;ln);o+:8+ln;i+:1]}

/ BPE encode
bpestep:{[tk]bs:-1e38;bi:-1;bx:-1;i:0
 while[i<(#tk)-1;mg:vocab[tk i],vocab[tk i+1];id:vocab?mg;if[(id<#vocab)&scores[id]>bs;bs:scores id;bi:id;bx:i];i+:1]
 $[bx<0;tk;((bx#tk),bi),(bx+2)_tk]}
bpe:{[t]tk:vocab?/:,:'t;$[|/tk=#vocab;'"char not in vocab";bpestep/[tk]]}

/ Core ops
rms:{[w;x]r:1%sqrt 1e-5+(+/x*x)%#x;`real$w*r*x}
mmf:{[w;x]`real$w$`float$x}                      / 2x memory bandwidth!
softmax:{e:exp x-|/x;`real$e%+/e}
silu:{`real$x%1+exp[-x]}

/ State
inist:{x::dim#0e;xb::dim#0e;xb2::dim#0e;hb::hdim#0e;hb2::hdim#0e
 q::dim#0e;k::dim#0e;v::dim#0e;att::(nh*sl)#0e;lg::vsz#0e
 kc::(nl*sl*dim)#0e;vc::(nl*sl*dim)#0e}

/ RoPE
rope:{[pos]idx:2*!dim div 2;fi:(pos*hs div 2)+(md[idx;hs])div 2
 fc:fcr fi;fs:fci fi;q0:q idx;q1:q idx+1;k0:k idx;k1:k idx+1
 q::`real$,/((q0*fc)-q1*fs),'(q0*fs)+q1*fc;k::`real$,/((k0*fc)-k1*fs),'(k0*fs)+k1*fc}

/ Attention for one head
attn:{[loff;pos;h]qh:hs#(h*hs)_q;ho:h*hs;np:1+pos
 idx:,/((!np)*dim)+\:loff+ho+!hs;ky:cut[hs;kc idx];sc:`real$(+/'ky*\:qh)%sqrt hs
 e:exp sc-|/sc;at:`real$e%+/e;vl:cut[hs;vc idx];xb[ho+!hs]::`real$+/at*vl}

/ One layer
layer:{[pos;l]xb::rms[rmsA l;x];q::mmf[wq l;xb];k::mmf[wk l;xb];v::mmf[wv l;xb]
 rope pos;loff:l*sl*dim;kvo:loff+pos*dim;kc[kvo+!dim]::k;vc[kvo+!dim]::v
 attn[loff;pos]'!nh;xb2::mmf[wo l;xb];x::x+xb2
 xb::rms[rmsF l;x];hb::mmf[w1 l;xb];hb2::mmf[w3 l;xb]
 hb::silu[hb]*hb2;xb::mmf[w2 l;hb];x::x+xb}

/ Forward
fwd:{[tok;pos]x::dim#(tok*dim)_emb;layer[pos]'!nl;x::rms[rmsO;x];lg::mmf[wcls;x]}

/ Sampling
amax:{*&x=|/x}
samp:{[lg]rv:rf32[]*+/lg;i:0;cp:0e;while[i<vsz;cp+:lg i;$[rv<cp;:i;i+:1]];0}
sampp:{[lg;tp]ix:>lg;pr:lg ix;cp:(+\)pr;co:*&cp>tp;$[^co;co:#pr;co+:1]
 pr:co#pr;ix:co#ix;rv:rf32[]*+/pr;i:0;cp:0e;while[i<co;cp+:pr i;$[rv<cp;:ix i;i+:1]];*ix}
sampn:{[tmp;tp]if[tmp=0e;:amax lg];l:softmax lg%tmp;$[(tp<1)&tp>0;sampp[l;tp];samp l]}

/ Generation
gen:{[ckpt;tokp;steps;tmp;topp;sd;prompt]
 -1"Loading...";rdcfg ckpt;-1"dim=",($dim)," layers=",($nl)," heads=",($nh)," vocab=",$vsz
 rdw ckpt;-1"Loaded weights";rdtok[tokp;vsz];-1"Tokenizer: ",$#vocab
 seed::$[sd=0;`long$.z.t;sd];if[(steps<1)|steps>sl;steps:sl]
 ptok:$[0<#prompt;bpe prompt;`int$()];-1"Prompt tokens: ",$#ptok;-1""
 inist[];tk:1;pos:0;t0:0;out:""
 while[pos<steps;fwd[tk;pos];nxt:$[pos<#ptok;ptok pos;sampn[tmp;topp]];pos+:1
  if[nxt=1;-1"";-1"tok/s: ",$(pos-1)%(`long$.z.p-t0)%1e9;:out]
  ts:vocab nxt;if[(tk=1)&0<#ts;if[ts[0]=" ";ts:1_ts]];1 ts;out,:ts;tk:nxt;if[t0=0;t0:.z.p]]
 -1"";-1"tok/s: ",$(pos-1)%(`long$.z.p-t0)%1e9;out}

/ CLI
usage:{-2"Usage: q golf.k -checkpoint <model.bin> [opts]";-2"-tokenizer -temp -topp -seed -steps -prompt";exit 1}
main:{a:.Q.opt .z.x;$[`checkpoint in!a;;usage[]];ck:*a`checkpoint;$[0=#ck;usage[];]
 tp:$[`tokenizer in!a;*a`tokenizer;"tokenizer.bin"]
 tmp:$[`temp in!a;"F"$*a`temp;1.0];topp:$[`topp in!a;"F"$*a`topp;0.9]
 sd:$[`seed in!a;"J"$*a`seed;0j];st:$[`steps in!a;"I"$*a`steps;256i]
 pr:$[`prompt in!a;*a`prompt;""];gen[`$":",ck;`$":",tp;st;tmp;topp;sd;pr];exit 0}
$[#.z.x;main[];]
