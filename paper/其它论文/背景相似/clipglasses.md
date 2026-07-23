TheFortiethAAAIConferenceonArtificialIntelligence(AAAI-26)
Not Just What’s There: Enabling CLIP to Comprehend Negated Visual
|     |     |     |     | Descriptions |     | Without | Fine-tuning |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------ | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- |
JunhaoXiao1,ZhiyuWu2,HaoLin3,YiChen1*,YahuiLiu4*,
XiaoranZhao5,ZixuWang5,ZejiangHe5
1CentralChinaNormalUniversity,
2FudanUniversity,
3HuazhongUniversityofScienceandTechnology,
4KuaishouTechnology,
5NationalUniversityofDefenseTechnology
chenyi30@mail.ccnu.edu.cn,yahui.cvrs@gmail.com
|     |     |     | Abstract |     |     |     |                                 |     | Before wearing glasses |                                 |     |     |     |
| --- | --- | --- | -------- | --- | --- | --- | ------------------------------- | --- | ---------------------- | ------------------------------- | --- | --- | --- |
|     |     |     |          |     |     |     | A picture with a girlbut no dog |     |                        | A picture with a girlbut no dog |     |     |     |
Vision-LanguageModels(VLMs)likeCLIPstruggletoun-
| derstand | negation, | often | embedding | affirmatives | and | nega- |     |     |     |     |     |     |     |
| -------- | --------- | ----- | --------- | ------------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
tivessimilarly(e.g.,matching“nodog”withdogimages).Ex-
istingmethodsrefinenegationunderstandingviafine-tuning
CLIP’stextencoder,riskingoverfitting.Inthiswork,wepro-
|                   |     |     |               |           |      |     | Mismatch×         | Match× |     |     | Match                | Match |     |
| ----------------- | --- | --- | ------------- | --------- | ---- | --- | ----------------- | ------ | --- | --- | -------------------- | ----- | --- |
| pose CLIPGLASSES, |     | a   | plug-and-play | framework | that | en- |                   |        |     |     |                      |       |     |
|                   |     |     |               |           |      |     | Negative Sentence |        |     |     | Affirmative Sentence |       |     |
hancesCLIP’sabilitytocomprehendnegatedvisualdescrip-
tions. CLIPGLASSES adapts a dual-stage design: a Lens A picture with a girlbut no dog A picture with a girlbut no dog
| module          | disentangles | negated       | semantics | from              | text embed- |        |     |     |     |     |     |     |     |
| --------------- | ------------ | ------------- | --------- | ----------------- | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- |
| dings, and      | a Frame      | module        | predicts  | context-aware     |             | repul- |     |     |     |     |     |     |     |
| sion strength,  | which        | is integrated |           | into the modified | similar-    |        |     |     |     |     |     |     |     |
| ity computation |              | to penalize   | alignment | with negated      | seman-      |        |     |     |     |     |     |     |     |
tics, thereby reducing false positive matches. Experiments Match Mismatch Match Match
showthatCLIPequippedwithCLIPGLASSESachievescom- After wearing glasses
petitivein-domainperformanceandoutperformsstate-of-the-
artmethodsincross-domaingeneralization.Itssuperiorityis Figure 1: CLIPGLASSES enhances CLIP’s capacity for
especiallyevidentunderlow-resourceconditions,indicating negationunderstandingbyintroducingadynamicrepulsion
strongerrobustnessacrossdomains. mechanismthatsuppressesimage-textsimilarityfornegated
|     |     |     |     |     |     |     | concepts, | thus enabling |     | inverse | matching | while | preserving |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------------- | --- | ------- | -------- | ----- | ---------- |
alignmentwithaffirmedcontent.
Code—https://github.com/Codecode-X/CLIPGlasses.git
Introduction Several existing approaches have attempted to adapt
|                 |     |                |     |                  |          |     | CLIP models | to  | negation-sensitive |     | tasks | through | parame- |
| --------------- | --- | -------------- | --- | ---------------- | -------- | --- | ----------- | --- | ------------------ | --- | ----- | ------- | ------- |
| Recent advances |     | in large-scale |     | pretraining have | advanced |     |             |     |                    |     |       |         |         |
Vision-Language Models (VLMs), with CLIP (Radford ter fine-tuning (Yuksekgonul et al. 2022; Alhamoud et al.
|     |     |     |     |     |     |     | 2025; Park | et al. | 2025; | Singh | et al. 2025), | but | these ap- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ----- | ----- | ------------- | --- | --------- |
etal.2021)emergingasafoundationalmodel.CLIPenables
|     |     |     |     |     |     |     | proaches | poses two | critical | drawbacks. |     | First, constructing |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | -------- | ---------- | --- | ------------------- | --- |
cross-modalalignmentbyprojectingimagesandtextsintoa
|                  |     |        |              |      |       |         | large-scale             | negation-annotated |     |         | datasets    | is time-consuming |     |
| ---------------- | --- | ------ | ------------ | ---- | ----- | ------- | ----------------------- | ------------------ | --- | ------- | ----------- | ----------------- | --- |
| shared embedding |     | space, | underpinning | core | tasks | such as |                         |                    |     |         |             |                   |     |
|                  |     |        |              |      |       |         | and resource-intensive. |                    |     | Second, | fine-tuning | introduces        | the |
retrieval,captioning,andtext-conditionedgeneration.
However, CLIP exhibits notable limitations in modeling risk of catastrophic forgetting, whereby enhanced negation
understandingcomesattheexpenseofdeterioratinggeneral-
| negation semantics. |     | As shown |     | in Figure 1, | CLIP | fails to |                      |     |      |            |     |             |        |
| ------------------- | --- | -------- | --- | ------------ | ---- | -------- | -------------------- | --- | ---- | ---------- | --- | ----------- | ------ |
|                     |     |          |     |              |      |          | purpose performance. |     | This | represents | a   | fundamental | trade- |
properlyhandlenegationcueslike“no”or“without”intext
offbetweenspecializedcapabilityandbroadapplicability.
| inputs, incorrectly |         | matching | negated | concepts         | with     | corre- |                       |       |              |           |             |             |          |
| ------------------- | ------- | -------- | ------- | ---------------- | -------- | ------ | --------------------- | ----- | ------------ | --------- | ----------- | ----------- | -------- |
|                     |         |          |         |                  |          |        | To address            | these | limitations, |           | we draw     | inspiration | from     |
| sponding visual     | content |          | rather  | than recognizing | their    | ab-    |                       |       |              |           |             |             |          |
|                     |         |          |         |                  |          |        | two key observations. |       | First,       | although  | affirmative |             | and neg- |
| sence. This         | failure | stems    | from    | the sparsity of  | negation | ex-    |                       |       |              |           |             |             |          |
|                     |         |          |         |                  |          |        | ative semantics       | lie   | close        | in CLIP’s | feature     | space,      | visual-  |
pressionsinpretrainingcorpora(i.e.,only0.7%(Parketal.
|     |     |     |     |     |     |     | ization analysis | (Figure |     | 2) reveals | structured | separability |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ------- | --- | ---------- | ---------- | ------------ | --- |
2025)),whichpreventscontrastivelearningfromeffectively
enabledbylayer-specificencoding(Quantmeyer,Mosteiro,
capturingsemanticpolarityreversals.
|     |     |     |     |     |     |     | and Gatt         | 2024), indicating |     | the  | feasibility  | of disentangling |          |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ----------------- | --- | ---- | ------------ | ---------------- | -------- |
|     |     |     |     |     |     |     | negation-related | information       |     | from | CLIP-encoded |                  | text em- |
*Correspondingauthors.
Copyright©2026,AssociationfortheAdvancementofArtificial beddings.Second,cognitivestudiesshowthathumanspro-
Intelligence(www.aaai.org).Allrightsreserved. cessnegationintwostages,firstidentifyingthenegatedcon-
10978

cept,theninvertingitsmeaning(Kaup,Zwaan,andLu¨dtke 2025b; Li et al. 2025a), captioning (Kim et al. 2025; Nam
2007;Orenes,Beltra´n,andSantamar´ıa2014;Zuanazzietal. et al. 2025; Li et al. 2025c) and VQA (Xing et al. 2024;
2024). Guided by these insights, we design a plug-and- Sun et al. 2024; Jiang et al. 2025) and generation (Zhang
playframeworkCLIPGLASSESthatleveragesCLIP’slatent etal.2025d;Fengetal.2025;Rombachetal.2022;Zhang
negation representations and emulates this two-stage pro- et al. 2025c; Fang et al. 2025; Zhang et al. 2025e), lever-
cess. agingrepresentationslearnedfromlarge-scalenoisyimage-
Specifically, CLIPGLASSES extends CLIP through two textpairs.However,theyexhibitcriticaldeficienciesinfine-
lightweightmodules.TheLensmoduledisentanglesnegated grainedsemanticunderstanding,manifestinginfourkeyar-
semantics from text embeddings, while the Frame module eas: over-reliance on shallow statistical cues (Zhang et al.
predicts context-dependent repulsion strength. These com- 2023; Geirhos et al. 2020), failures in compositional rea-
ponents jointly enable a modified similarity computation: soning(Yuksekgonuletal.2022;Thrushetal.2022),weak
negated content identified by Lens is explicitly penalized attribute grounding, and bag-of-words-like text processing
through repulsion vectors whose magnitude is dynamically that ignores syntactic structure (Koishigarina, Uselis, and
scaledbyFrame’sstrengthpredictions.Thisintegratedpro- Oh2025).Negationunderstandingposesaparticularlychal-
cessisvisuallysummarizedinFigure3.Tocoordinatethese lenging case, with VLMs frequently failing to distinguish
components, we employ a progressive three-stage regimen negated and affirmative inputs (Morante and Blanco 2021;
with frozen CLIP parameters: first training Lens to dis- Ma et al. 2023) due to affirmation bias and scarcity of
entangle negated text representations, then training Frame negation in pretraining data (occurring in ≤0.7% of cap-
to model dynamic repulsion using ground-truth negation tions (Park et al. 2025)), significantly impairing negation-
features, and finally joint optimization of both modules sensitive applications, such as medical or clinical scenar-
with Lens outputs to enhance synergy. In this way, instead ios(KoandPark2025;Suetal.2025).
of modifying the model’s “eyes”, we simply let it wear Prior Work on Negation Understanding. Existing meth-
“glasses”tobetterperceivethenegation. ods universally fine-tune CLIP’s text encoder and fall into
ExperimentalresultsestablishCLIPGLASSES’sbalanced two categories based on data strategy: (1) Compositional
advantages through systematic comparisons. While fine- Perturbation, typified by NegCLIP (Yuksekgonul et al.
tuning baselines overfit negation datasets for marginal in- 2022), which uses word-order shuffling to simulate syntac-
domain gains (CoN-CLIP: 99.70% vs. our 96.56% on tic variation in negation; and (2) Paraphrastic Augmenta-
CC-Neg-val), this incurs unwilling costs: degraded cross- tion, including CoN-CLIP (Singh et al. 2025), Negation-
domain generalization (25.70% vs. our 34.51% on Neg- CLIP (Park et al. 2025), and NegBench (Alhamoud et al.
COCO-MCQ).Underlow-resourceconstraints(5Kimages), 2025), which leverage LLM-generated negatives to enrich
these limitations intensify—our approach surpasses CoN- contrastivetraining.Whileeffectivein-domain,thesemeth-
CLIP by 27.45 points on CC-Neg-val and 5.29 points on ods overwrite CLIP’s pretrained representations, risking
Neg-COCO-MCQ.Moreover,unlikefine-tuningapproaches overfitting, catastrophic forgetting (Zhai et al. 2024; Jha,
thatimpairCLIP’snativezero-shotperformanceonstandard Gong,andYao2024;Lietal.2024),andpoorcross-domain
non-negationbenchmarks,ournon-invasivearchitectureef- generalizationduetothelackofexplicitnegationmodeling.
fectivelypreservestheseintrinsiccapabilities.Theseresults In contrast, CLIPGLASSES introduces dedicated negation
substantiate the efficacy of CLIPGLASSES in achieving reasoning via architectural extensions, preserving CLIP’s
transferable,semanticallygroundednegationunderstanding zero-shotcapabilitieswithoutmodifyingitsparameters.
without sacrificing model robustness or generalization. We
summarizeourkeycontributionsasfollows: Preliminary:Two-StageModelingfor
• We present CLIPGLASSES, a non-intrusive framework NegationUnderstanding
enhancingCLIP’snegationmodelingviahuman-inspired
two-stageprocessingwithoutparametermodification. Although affirmative and negative semantics are often em-
bedded in close proximity within CLIP’s feature space, vi-
• We design a novel architecture, including a syntax-
sualization analysis (See Figure 2) reveals a promising po-
semanticLensfordisentanglingnegationsemantics,and
tentialforsemanticdisentanglement.Thisseparabilityarises
a Frame for modeling context-aware repulsion, and a
fromlayer-specificnegationencodingmechanisms(Quant-
modified similarity computation that explicitly reverses
meyer,Mosteiro,andGatt2024)andformsasolidbasisfor
alignmentwithnegatedcontent.
targetedextractionofnegation-relatedinformation.
• Ourmethodattainsstate-of-the-arttrade-offsbetweenin-
Additionally, findings from cognitive neuroscience indi-
domainaccuracyandcross-domaingeneralization,with-
cate that humans typically process negation in two distinct
outcompromisingCLIP’snativezero-shotabilities.
stages(Kaup,Zwaan,andLu¨dtke2007;Orenes,Beltra´n,and
Santamar´ıa2014;Zuanazzietal.2024):First,byidentifying
RelatedWork
the target object or concept being negated; Second, by in-
Vision-LanguageModelsandTheirLimitations.Vision- vertingitssemanticimplicationtoderivethenegatedmean-
language models like CLIP (Radford et al. 2021) demon- ing. Inspired by this two-stage cognitive mechanism, we
strate strong performance on broad cross-modal tasks, in- proposeacorrespondingcomputationalmodelingstrategyto
cludingretrieval(Qinetal.2025;Caffagnietal.2025;Fang betterhandlenegationinvision-languagealignment.Inthe
etal.2024;Zhangetal.2025a;Lietal.2025b;Zhangetal. firststage,themodelexplicitlyidentifiesandextractsnega-
10979

t-SNE visualization extract features T ,T ,T from the first three layers of the
1 2 3
CLIP text encoder,which encode low-level syntactic infor-
mation(Quantmeyer,Mosteiro,andGatt2024).Eachlayer’s
featuresareprojectedintoasharedlatentspaceviathetrans-
formation:
P(X)=LN(GELU(WX)) (1)
whereW denotesalearnableweightmatrixuniquetoeach
projection instance, GELU is the Gaussian Error Linear
Unit (Hendrycks and Gimpel 2016), and LN represents
LayerNormalization(Ba,Kiros,andHinton2016).There-
sultingqueryvectorsare:
Q =P (T ), i∈{1,2,3} (2)
i q,i i
Semantic Stream. While syntactic structures signal nega-
tion expression, precise interpretation requires global se-
mantic comprehension. For instance, in the sentence “He
Figure 2: t-SNE (Cai and Ma 2022) visualization of CLIP
claimedhefinishedthework,butactuallyhadn’t”,thenega-
text features for multiple positive-negative sentence pairs
tion scope of “hadn’t” depends critically on the antecedent
(e.g., “there is a woman” vs. “there is not a woman”). Cir-
context(“finishedthework”)andthecontrastiveconjunction
clesandsquaresdenotepositiveandnegativeforms,colors
(“but”).Tocapturesuchcontext-dependentnegation,weuti-
distinguish different pairs. Feature clusters across pairs are
lizethefinal-layeroutputT oftheCLIPtextencoder(Lin
well-separated, showing CLIP’s strong instance-level dis- clip
et al. 2024; Jing et al. 2024), which provides globally con-
crimination.However,positiveandnegativefeatureswithin
textualizedrepresentationsforgenerating:
individual pair remain closely positioned, indicating that
whileCLIPhaslimitednegationmodelingcapabilities,there K =P (T ), V =P (T ), i∈{1,2,3} (3)
k clip i v,i clip
existsclearpotentialforsemanticdisentanglement.
Hierarchical Attention Fusion. Syntactic representations
from different layers encode linguistic features at vary-
tionsemanticsfromtextualfeatures.Inthesecondstage,it inggranularities:Q 1 captureslocaltokeninteractions(e.g.,
modulates the image-text similarity by explicitly suppress-
negationparticlesandadjacentverbs),whilethedeeperQ
3
ingalignmentwithnegatedconcepts. encodes phrasal-level dependencies. To dynamically inte-
grate these hierarchical patterns with global semantics, we
Methodology computeattentionweightsbetweeneachQ i andtheseman-
tickeyK,modulatedbylearnablescalarsα thatadaptively
i
To instantiate the two-stage modeling paradigm outlined
calibrateeachsyntacticlevel’scontribution:
above,wepropose CLIPGLASSES,amodularextensionto
C pr L is I e P s fo tw r o ne c g o a m tio p n o - n a e w n a ts re : v L i e s n io s n , - w la h n i g c u h ag d e is a e l n i t g a n n m gl e e n s t. n I e t g c a o t m ed - T attn = X 3 softmax (cid:18) Q √i K⊤ +logα i (cid:19) V i (4)
D
semantics from text embeddings, and Frame, which pre- i=1
dicts context-dependent repulsion strength. These compo- Residual Gating. While hierarchical attention enriches
nents are integrated into a modified similarity computation the representation with negation-relevant structure, over-
thatpenalizesalignmentwithnegatedcontent.Anoverview relianceonitmaycausesemanticdriftorlossofcorecon-
ofthearchitectureisshowninFigure3.Wedescribethear- tent.Toaddressthis,weapplyaresidualgatethatadaptively
chitectureandfunctionalityofeachmoduleinthefollowing blends the attended representation with the original CLIP
andthenpresentthefinalmatchingstrategy. features:
g =σ(W T +b ) (5)
g attn g
Lens:Syntax–SemanticDual-StreamArchitecture
(cid:0) (cid:1)
T =FFN g⊙T +(1−g)⊙T (6)
WhileCLIP’stextencodereffectivelymodelsaffirmativese- neg attn clip
mantics,itlacksexplicitmechanismsforcapturingthestruc- where σ is the sigmoid and ⊙ denotes Hadamard multipli-
turalandcontextualcharacteristicsofnegation.Asaresult, cation.Thegategactsasasoftselector,allowingthemodel
itsoutputfeaturesoftenfailtodistinguishthesemanticdif- toamplifystructuraladjustmentsonlywhennecessary.The
ferences inherent in negated expressions. To address this bias b g is initialized to a negative value to favor original
limitation,wepropose theLensmodule,a syntax-semantic features during early training. Finally, a feed-forward neu-
dual-stream architecture that explicitly models negation- ralnetwork(FFN)refinestheoutputrepresentation.
awaretextrepresentationsviahierarchicalattention.
Frame:Cross-ModalDynamicRepulsionWeight
Syntactic Stream. Natural language negation frequently
Generator
manifests through syntactic patterns (e.g., auxiliary con-
structions like “do not”, adverbials like “never”) that ex- Negationinlanguagevariesinintensityandcontextualnu-
hibitlocalstructuraldependencies.Tocapturethesecues,we ance(e.g.,“not”vs.“maynot”),directlyaffectingthedegree
10980

Syntax
𝑇 𝑇 𝑇 1 2 3
𝑇 1
A picture Semantic Lens 𝑇 2
with a girl Text 𝑇 clip 𝑇 𝑛𝑒𝑔<dog> 𝑇 3 𝑇 𝑛𝑒𝑔
but no dog. Encoder <girl><dog>
Image 𝐼 clip 𝝀
Encoder
Frame
ሼ
CLIP GLASSES
A picture with a girlbut no dog Step1 𝑇 Step2
𝑇 𝑐𝑙𝑖𝑝 <girl><dog>… 𝑇 𝑛𝑒𝑔 <dog> 𝐼 𝑐𝑙𝑖𝑝 𝑇 𝑛 𝑐𝑙 𝑒 𝑖 𝑔 𝑝 𝝀 𝑺 𝑇 𝐼 𝑐 𝑐 𝑙 𝑙 𝑖 𝑖 𝑝 𝑝 𝝀𝑺 𝑇 𝐼 𝑐 𝑛 𝑙 𝑒 𝑖 𝑔 𝑝 𝑺 𝑇 𝐼
Lens Image-Text Matching
Syntax
𝑸 𝟏 𝑇 clip
𝑸 𝟐 𝑲
<girl><dog>
𝑸 𝟑
Early Layers 𝑽 𝟏 𝑽 𝟐 𝑽 𝟑 Residual Gate 𝐼 𝑐𝑙𝑖𝑝
Text 𝑇 clip 𝝀
Encoder <dog>
Semantic
𝑇 𝑛𝑒𝑔
Legend
Align
Repel
Overview
Input Text
Input Image
Frame Repulsion 𝑽 Active on negated only
Weight 𝑇 𝑛𝑒𝑔 C
𝑲
𝓜∈ 0,1
𝑸
𝑇 𝑖 Layer iText Emb. Cross Attention I E m n a co g d e er 𝐼 clip C C r o o n s t s e - x M t odal C N l e a g s a si t f io ie n r
𝐼 clipImage Emb. 𝑇 clip Text Emb. Self Attention G
𝑇 clip C 𝝀 𝑇 clip
Add Sigmoid G GELU C Concat FFN Linear
Figure 3: CLIPGLASSES enhances CLIP’s capability to model negative semantics by introducing two modules: Lens and
Frame.Lensdisentanglesnegatedconcepts(e.g.,“dog”in“nodog”)fromthetextembeddingT .Framedynamicallypredicts
clip
arepulsionstrengthλbasedoncross-modalcontext.ThefinalsimilarityscoreiscomputedasS = S −λ·Sneg,aligning
I2T I2T
imageswithaffirmedcontentwhilerepellingfromnegatedconceptswhennegationispresentinthetext.
towhichnegatedconceptsinfluencevision-languagealign- text-side output F(1) from the joint self-attention. To pre-
T&I
ment. To address this, the Frame module dynamically esti- serveoriginalsemanticswhileenablingadaptivefusion,we
matesarepulsionweightλbasedonjointimage-textrepre- applyaresidualconnection:
sentations,whichgovernsthestrengthofsemanticinversion.
Cross-ModalContext.Negationisinherentlylinguistic,yet T fuse =αTˆ clip +F T (1 & ) I (9)
its interpretation is often grounded in visual context (Sun
whereα∈[0,1]isalearnablescalar(initializedto0.1)that
etal.2021;Janssensetal.2024).Toestimateλinacontext-
controls the balance between the original and the context-
aware manner, we first encode cross-modal interactions by
enhancedtextfeatures.
allowingeachmodalitytoattendtotheother.
Dynamic Repulsion Weight. Estimating the repulsion
Before fusing the textual and visual features, we apply
strengthλhingesonhighlightingnegatedsemanticsrelevant
L2normalizationtoeliminatescalediscrepanciesthatcould
tothefusedtext-visualcontext.Sincenegationinherentlyin-
hinder effective attention. For numerical stability, a small
volvescontrastingtheoriginalsemanticswithnegatedcon-
constant ϵ is added in the denominator when normalizing
cepts,amechanismthatattendstothenegatedfeaturescon-
thenegatedtextfeature:
ditionedontheenrichedtextualcontextisessential.
Iˆ = I clip , Tˆ = T clip , Tˆ = T neg Tothisend,wefirstcomputeacross-attentionoutputC,
clip ∥I ∥ clip ∥T ∥ neg ∥T ∥ +ϵ where the fused representation T acts as the query, and
clip 2 clip 2 neg 2 fuse
(7) thenegatedsemanticsTˆ serveaskeysandvalues.Thisal-
neg
To capture bidirectional cross-modal dependencies, we lowsthemodeltodynamicallyalignandweightthenegated
employ a joint self-attention mechanism (Vaswani et al. informationrelevanttothecurrentcontext:
2017)overconcatenatedtextandimagefeatures.Compared
C =CrossAttn(T , Tˆ , Tˆ ) (10)
to directional cross-attention, this symmetric early fusion fuse neg neg
|{z} |{z} |{z}
enablesbothmodalitiestoattendtoeachotherjointly,avoid-
Q K V
ingrepresentationalbias.Formally:
Next, the repulsion weight λ is estimated by projecting
(cid:16) (cid:17)
F =SelfAttn [Tˆ ;Iˆ ] ∈R2×D (8) the concatenation of the fused feature T and the cross-
T&I clip clip fuse
attentionoutputC throughalearnablelinearlayerfollowed
Sinceλmodulatestextualsemantics,itsestimationshould
byasigmoidactivation:
begroundedinarepresentationthatistext-centricyetcon-
textually enriched by visual cues. We thus retain only the λ=σ(W [T ;C]+b ) (11)
λ fuse λ
10981

Image-TextMatching Toensurestableoptimization,themaskisfixedtoM=1
duringtraining,allowinggradientstopropagateconsistently.
| With the | negated | text representation |     | T   | generated | by the |     |     |     |     |     |     |     |     |
| -------- | ------- | ------------------- | --- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
neg
Lens module and the context-sensitive repulsion weight λ Itisonlyactivatedatinferencetimetoselectivelysuppress
estimatedbytheFramemodule,wenowdescribehowthese negatedalignments.
componentsjointlycontributetothefinalimage-textmatch-
TrainingStrategy
ingobjective.Thecoreideaistopreservetheoriginalalign-
mentabilityofCLIPwhilepenalizingfalsepositivescaused We adopt a staged training strategy to progressively model
byvisual-textualagreementwithnegatedsemantics. cross-modal negation semantics. The process consists of
Thefinalsimilarityscorethuscombinesastandardmatch- three phases: (1) training the Lens module to capture fine-
| ing term | with a | negation-aware |     | repulsion | component. | For |         |                 |     |     |         |       |     |              |
| -------- | ------ | -------------- | --- | --------- | ---------- | --- | ------- | --------------- | --- | --- | ------- | ----- | --- | ------------ |
|          |        |                |     |           |            |     | grained | representations |     | of  | negated | text; | (2) | training the |
generality across both text-to-image and image-to-text re- Frame module to establish dynamic interactions between
trieval, we adopt a unified notation where q denotes the negation and image features; and (3) joint optimization to
querymodality(eithertextorimage)andk denotesthekey enhancecollaborationbetweenthetwomodules.
modality(thecorrespondingpairedinstance). Stage 1: Independent Training of the Lens Model. In
| Basic Similarity |     | Calculation. |     | We begin | with the | original |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------------ | --- | -------- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
thisstage,weextractthenegatedobjectfromtheinputtext
CLIPsimilarityasthebasescore.Thistermreflectsthese- and generate a short prompt (e.g., “This image shows a
manticalignmentbetweentheimageandthetext: {negobj}”).ThispromptisthenpassedtotheCLIPtexten-
|     |     |     |     |     |     |     | coder | to extract | the | ground | truth | negation | feature | Tneg ∈ |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | --- | ------ | ----- | -------- | ------- | ------ |
|     |     |     |     |     | q⊤k |     |       |            |     |        |       |          |         | obj    |
RD
|     | S base | (q,k)=exp(θ |     | T )· |     | (12) | asthesupervisionsignal.                         |     |     |     |     |     |     |     |
| --- | ------ | ----------- | --- | ---- | --- | ---- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|     |        |             |     | ∥q∥  | ∥k∥ |      |                                                 |     |     |     |     |     |     |     |
|     |        |             |     |      | 2 2 |      | Wedesignatrainingobjectivethatcombinestheseman- |     |     |     |     |     |     |     |
where θ is the temperature parameter of the pretrained tic similarity loss L and the cross-modal alignment loss
|            | T   |     |     |     |     |     |     |                                                |     | sim |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- |
| CLIPmodel. |     |     |     |     |     |     | L   | toensurethelearnednegationfeaturesarebothaccu- |     |     |     |     |     |     |
align
| Negation-aware |     | Similarity | Adjustment. |     | While | the base |     |     |     |     |     |     |     |     |
| -------------- | --- | ---------- | ----------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
rateandalignedwithimagefeatures:
similaritycapturesstandardsemanticalignment,itisinsuf-
|     |     |     |     |     |     |     |     |     |     | L=L | +δL |     |     | (16) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
ficient for handling negation. If the text explicitly negates sim align
| certain                                             | visual content, |     | the model | must | reduce | the match- |                                             |          |            |     |        |         |          |        |
| --------------------------------------------------- | --------------- | --- | --------- | ---- | ------ | ---------- | ------------------------------------------- | -------- | ---------- | --- | ------ | ------- | -------- | ------ |
|                                                     |                 |     |           |      |        |            | The                                         | semantic | similarity |     | loss L | ensures | semantic | con-   |
| ingscoreaccordingly.Tothisend,weintroduceanegation- |                 |     |           |      |        |            |                                             |          |            |     | sim    |         |          |        |
|                                                     |                 |     |           |      |        |            | sistencybetweenthepredictednegationfeatureT |          |            |     |        |         |          | andthe |
| awarerepulsiontermthatreflectshowwelltheimagealigns |                 |     |           |      |        |            |                                             |          |            |     |        |         |          | neg    |
groundtruthnegationfeatureTneg:
obj
withthenegatedsemantics.Therepulsiontermiscomputed
| by: |        |          |     |        |       |          |     |       |     |     |       | ( j )   | n e g , ( j )  |      |
| --- | ------ | -------- | --- | ------ | ----- | -------- | --- | ----- | --- | --- | ----- | ------- | -------------- | ---- |
|     |        |          |     |        |       |          |     |       |     | 1 B | T     | · T     |                |      |
|     |        | (cid:18) |     |        |       | (cid:19) |     |       |     | X   | n     | e g o   | b j            |      |
|     |        |          |     | q⊤k    |       |          |     | L sim | =1− |     |       |         |                | (17) |
| R   | =λ·max | exp(θ    |     | )·     | neg , | 0        |     |       |     | B   | ∥T (j | ) ∥ ∥ T | n e g , ( j )∥ |      |
| neg |        |          | T   |        |       | (13)     |     |       |     | j=1 | neg   | 2       |                | 2    |
|     |        |          |     | ∥q∥ ∥k | ∥     |          |     |       |     |     |       |         | obj            |      |
|     |        |          |     | 2      | neg 2 |          |     |       |     |     |       |         |                |      |
L
where the max(·,0) operation ensures that the repulsion The cross-modal alignment loss align constrains the
term only penalizes positive alignment with negated con- alignmentbetweenthenegationfeatureT neg andtheimage
featureI,preventingsemanticdrift:
cepts,preventingundesirablescoreinflation.
F in a l S im i la r it y S c o r e. T o a v o i d u n n e ce s s a r y i n t e r f ere n c e (cid:12) n e g,(j)·I(j) (cid:12)
|     |     |     |     |     |     |     |     | 1   | B (cid:12) | T ( j ) ·I(j) |     |     | T   | (cid:12) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | --- | --- | -------- |
in a ffi r m ati v e c a se s a n d b et te r p r e se r v e C L I P ’ s p e r f o r m a n c e X (cid:12) n e g o b j (cid:12)
|     |     |     |     |     |     |     | L align | =   |     |     |     | −   |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
ongeneral(non-negated)tasks,wefurtherintroduceacon- B (cid:12) ( j )∥ ∥I(j)∥ n e g,(j)∥ ∥I(j)∥ (cid:12)
|     |     |     |     |     |     |     |     |     | (cid:12)∥T | n e g 2 | 2   | ∥T  |     | 2 2 (cid:12) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------- | --- | --- | --- | ------------ |
ditional mask that selectively activates the repulsion term j=1 o b j
(18)
basedonthepresenceofnegation. Additionally, we implement a dynamic loss balancing
| Concretely, | we         | define | a binary | decision       | variable | M ∈    |           |          |          |           |          |     |             |            |
| ----------- | ---------- | ------ | -------- | -------------- | -------- | ------ | --------- | -------- | -------- | --------- | -------- | --- | ----------- | ---------- |
|             |            |        |          |                |          |        | strategy, | starting | with     | a higher  | emphasis |     | on semantic | sim-       |
| {0,1} that  | determines |        | whether  | the similarity | score    | should |           |          |          |           |          |     |             |            |
|             |            |        |          |                |          |        | ilarity   | (δ =     | 0.5) and | gradually | shifting |     | weight      | toward the |
becorrected.Thedecisionismadebyalightweightnegation
|             |     |                                      |     |     |     |     | cross-modalconstraint(δ |     |     | =1.0)tostabilizenegationrepre- |     |     |     |     |
| ----------- | --- | ------------------------------------ | --- | --- | --- | --- | ----------------------- | --- | --- | ------------------------------ | --- | --- | --- | --- |
| classifierG | :RD | →[0,1],whichpredictstheprobabilityof |     |     |     |     |                         |     |     |                                |     |     |     |     |
sentationlearningbeforeenforcingimagealignment.
negationfromtheoriginaltextrepresentation: Stage2:IndependentTrainingoftheFrame.Inthisstage,
|     |     |     |     |     |     |     |                  |     |          |     |         | Tneg | ∈ RD |             |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | -------- | --- | ------- | ---- | ---- | ----------- |
|     |     |     |     |     |     |     | the ground-truth |     | negation |     | feature |      |      | is directly |
obj
M=I[G(T )>τ ] (14) usedasinputtotheFramemodule,servingasthepredicted
|     |     |     | clip | neg |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
negationfeaturefromtheLens.Trainingisbasedonagen-
| where τ | is a | threshold | (default | 0.5). | The classifier | G is |                      |     |     |     |     |     |     |     |
| ------- | ---- | --------- | -------- | ----- | -------------- | ---- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
|         | neg  |           |          |       |                |      | eralizedInfoNCEloss: |     |     |     |     |     |     |     |
implementedasatwo-layerMLP.
| The repulsion |     | term R | neg is | applied | only when | M = 1, |     |     |     |     |     |     |     |     |
| ------------- | --- | ------ | ------ | ------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
ensuring that the correction is restricted to negated inputs eS(q,k+)
(q,k+,N)=−log
| whilemaintainingstandardCLIPbehaviorelsewhere.       |     |     |     |     |     |     | L gen |     |     |     |           | P   |      |          |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --------- | --- | ---- | -------- |
|                                                      |     |     |     |     |     |     |       |     |     |     | eS(q,k+)+ |     |      | eS(q,k−) |
| ThefinalsimilarityscoreintegratesthebaseCLIPsimilar- |     |     |     |     |     |     |       |     |     |     |           |     | k−∈N |          |
(19)
itywiththenegation-awarerepulsiontermunderthecontrol
|                       |     |      |      |     |     |      | whereq                                    | denotesthequeryfeature,k+ |       |         |             |      | thecorresponding |           |
| --------------------- | --- | ---- | ---- | --- | --- | ---- | ----------------------------------------- | ------------------------- | ----- | ------- | ----------- | ---- | ---------------- | --------- |
| oftheconditionalmask: |     |      |      |     |     |      |                                           |                           |       |         |             |      | k−.              |           |
|                       |     |      |      |     |     |      | positive                                  | key,                      | and N | the set | of negative | keys |                  | The simi- |
|                       |     | S =S | −M·R |     |     | (15) | larityfunctionS(·,·)isdefinedasinEq.(15). |                           |       |         |             |      |                  |           |
|                       |     |      | base | neg |     |      |                                           |                           |       |         |             |      |                  |           |
10982

|        |     |             |     | TrainingSize      |     |     |            |     |          | Testing |            |     |     |          |     |
| ------ | --- | ----------- | --- | ----------------- | --- | --- | ---------- | --- | -------- | ------- | ---------- | --- | --- | -------- | --- |
| Method |     | TrainingSet |     |                   |     |     |            |     |          |         |            |     |     |          |     |
|        |     |             |     | (Images/Captions) |     |     | TestingSet |     | Accuracy |         | TestingSet |     |     | Accuracy |     |
NegCLIP COCO-caption 82K/31K CC-Neg-val 62.63% Neg-COCO-MCQ 10.20%
CoN-CLIP CC-Neg 188K/376K CC-Neg-val 99.70% Neg-COCO-MCQ 25.70%
CLIPGlasses CC-Neg 188K/376K CC-Neg-val 96.56%-3.14% Neg-COCO-MCQ 34.51%+8.81%
CoN-CLIP Neg-COCO-R 5K/22K CC-Neg-val 65.91% Neg-COCO-MCQ 30.61%
CLIPGlasses Neg-COCO-R 5K/22K CC-Neg-val 93.36%+27.45% Neg-COCO-MCQ 35.90%+5.29%
Table1:Performancecomparisonhighlightingthetrade-offbetweenin-domainfittingandcross-domaingeneralization.
Dependingontheavailabilityofhardnegatives,thelossis Method TrainingSet ImageNet caltech101
appliedintwoways.Whenexplicithardnegativesareavail-
|     |     |     |     |     |     |     |     | VanillaCLIP |     |     | -   |     | 53.87 | 90.96 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | ----- | ----- | --- |
able (e.g., from the CCNeg dataset), training is performed CoN-CLIP CC-Neg 50.98 88.91
(q,k+,k−).
using constructed triplets Otherwise, in-batch CLIPGLASSES Neg-COCO-R 53.51 90.54
contrastivelearningisadopted,treatingallotherbatchsam- CLIPGLASSES CC-Neg 53.28 90.97
plesasimplicitnegatives.
| Stage 3: | Joint | Training | of Lens | and | Frame. | After | train- |       |              |             |     |     |          |              |     |
| -------- | ----- | -------- | ------- | --- | ------ | ----- | ------ | ----- | ------------ | ----------- | --- | --- | -------- | ------------ | --- |
|          |       |          |         |     |        |       |        | Table | 2: Zero-shot | performance |     | on  | standard | non-negation |     |
ing the Lens and Frame separately, we further design a benchmarksformodelstrainedonnegationdatasets.
| joint training                       | process |     | to optimize  | their    | collaborative | effect.    |        |                                                    |     |     |     |     |     |     |     |
| ------------------------------------ | ------- | --- | ------------ | -------- | ------------- | ---------- | ------ | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| Specifically,                        | based   | on  | the training | strategy |               | of Stage   | 2, the |                                                    |     |     |     |     |     |     |     |
| ground-truthnegatedobjectfeatureTneg |         |     |              |          | ∈RD           | intheinput |        |                                                    |     |     |     |     |     |     |     |
|                                      |         |     |              |          | obj           |            |        | seeninCoN-CLIP’sdropunderconstraints,andtheabsence |     |     |     |     |     |     |     |
of the Frame is replaced with the output T neg of the Lens ofexplicitnegationmodeling,asreflectedinNegCLIP’sper-
module,whiletherestofthesettingsremainthesameasin sistentlylowperformance.Incontrast,ournon-invasivear-
Stage2. chitecturemaintainsconsistentperformanceacrosssettings,
validatingitsstrengthincapturingtransferable,semantically
|             | ComparativeExperiments |         |     |         |            |         |     | groundednegation. |     |     |     |     |     |     |     |
| ----------- | ---------------------- | ------- | --- | ------- | ---------- | ------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
| To validate | our                    | method, | we  | conduct | systematic | compar- |     |                   |     |     |     |     |     |     |     |
InherentAbilityRetentionAnalysis
| isons with | state-of-the-art |               | baselines, |     | evaluate | the retention |     |            |                   |     |           |          |               |      |         |
| ---------- | ---------------- | ------------- | ---------- | --- | -------- | ------------- | --- | ---------- | ----------------- | --- | --------- | -------- | ------------- | ---- | ------- |
|            |                  |               |            |     |          |               |     | A critical | concern           | in  | enhancing | negation | understanding |      | is      |
| of CLIP’s  | zero-shot        | capabilities, |            | and | perform  | ablations     | to  |            |                   |     |           |          |               |      |         |
|            |                  |               |            |     |          |               |     | whether    | such improvements |     |           | come at  | the cost      | of a | model’s |
assessthecontributionsofindividualcomponents.
|     |     |     |     |     |     |     |     | inherent    | strengths. | To       | assess       | this, | we compare | zero-shot |       |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | -------- | ------------ | ----- | ---------- | --------- | ----- |
|     |     |     |     |     |     |     |     | performance | on         | standard | non-negation |       | benchmarks |           | (Ima- |
ComparativeAnalysis
|     |     |     |     |     |     |     |     | geNet | (Deng et | al. 2009) | and | Caltech101 | (Li | et al. | 2022)) |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | -------- | --------- | --- | ---------- | --- | ------ | ------ |
Existingworktypicallyfine-tunesCLIP’stextencoderwith- aftertrainingonnegationdatasets.
outarchitecturalchanges.Giventheirstructuralhomogene-
AsshowninTable2,CLIPGLASSESretainsnear-original
ity—differing primarily in training data—we focus on two performance,matchingorevensurpassingthevanillaCLIP
fully open-source, paradigm-representative baselines: Neg- on both datasets. In contrast, the CoN-CLIP shows notable
CLIP(Yuksekgonuletal.2022)andCoN-CLIP(Singhetal. degradation, particularly on ImageNet. These results con-
2025).ComparativeresultsareshowninTable1. firm that our non-invasive design preserves CLIP’s general
TrainedontheCC-Negdataset(Singhetal.2025)(188K visual-languagealignmentabilitywhileenhancingnegation
| images,376Kcaptions),CLIPGlassesachieves96.56%ac- |     |     |     |     |     |     |     | understanding. |     |     |     |     |     |     |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
curacyonthein-domainCC-Neg-valset.Whileslightlybe-
low CoN-CLIP’s 99.70%, this reflects a deliberate design AblationAnalysis
choicetoprioritizegeneralizationoveroverfitting.Theben-
|     |     |     |     |     |     |     |     | To assess | the | individual | contributions |     | of key | components, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ---------- | ------------- | --- | ------ | ----------- | --- |
efitbecomesevidentonthecross-domainNeg-COCO-MCQ we conduct hierarchical ablation experiments on the CC-
benchmark(Alhamoudetal.2025),whereCLIPGlassessur- Negdataset.Usingthefullmodelasourreferencestandard,
passes CoN-CLIP by 8.81 percentage points (34.51% vs. we evaluate two key metrics: classification accuracy (Ac-
| 25.70%). |              |     |            |       |     |            |     | curacy)andtheaverageconfidencemarginbetweenpositive |     |     |     |     |     |     |     |
| -------- | ------------ | --- | ---------- | ----- | --- | ---------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| Under    | low-resource |     | conditions | using | the | Neg-COCO-R |     |                                                     |     |     |     |     |     |     |     |
andnegativepairs,termedfalsealignmentrate(FAR),which
| subset (5K     | images, | 22K     | captions; | Alhamoud    |     | et al. 2025), |     | isdefinedas: |     |     |     |     |     |     |     |
| -------------- | ------- | ------- | --------- | ----------- | --- | ------------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
| this advantage |         | becomes | more      | pronounced: |     | CLIPGlasses   |     |              |     |     |     |     |     |     |     |
N
outperforms CoN-CLIP by 27.45 points on CC-Neg-val 1 X(cid:12) (cid:12)s+−s−(cid:12)
|                       |             |     |        |      |        |              |     |     |     | FAR= |     |     | (cid:12) |     | (20) |
| --------------------- | ----------- | --- | ------ | ---- | ------ | ------------ | --- | --- | --- | ---- | --- | --- | -------- | --- | ---- |
| (93.36%               | vs. 65.91%) |     | and by | 5.29 | points | on Neg-COCO- |     |     |     |      | N   | i   | i        |     |      |
| MCQ(35.90%vs.30.61%). |             |     |        |      |        |              |     |     |     |      | i=1 |     |          |     |      |
|                       |             |     |        |      |        |              |     |     | s+  | s−   |     |     |          |     |      |
These results highlight two core limitations of fine- where and denote the similarity scores between the
|     |     |     |     |     |     |     |     |     | i   | i   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tuning-based approaches: reliance on large-scale data, as imageandthepositiveandnegativecaptionsinthei-thsam-
10983

AblationSetting Acc ∆Acc FAR ∆FAR Negation Strength Distribution (Ridge Plot)
| ————————————-LensModule———————————  |     |        |     |         |        |         |     | Min    |           |        |     |                        |     | Max    |
| ----------------------------------- | --- | ------ | --- | ------- | ------ | ------- | --- | ------ | --------- | ------ | --- | ---------------------- | --- | ------ |
| w/oSyntaxStream                     |     | 94.09% |     | -2.47%  | 4.4927 | -3.3326 |     |        |           | “no”   |     | 0.5599                 |     |        |
|                                     |     |        |     |         |        |         |     | 0.2628 |           |        |     |                        |     | 0.9574 |
| w/oSemanticStream                   |     | 94.93% |     | -1.63%  | 6.0925 | -1.7328 |     |        |           |        |     |                        |     |        |
| w/oResidualGating                   |     | 68.93% |     | -27.63% | 0.7875 | -7.0378 |     |        |           |        |     |                        |     |        |
|                                     |     |        |     |         |        |         |     |        | “not any” |        |     | 0.5359                 |     |        |
| ————————————-FrameModule——————————– |     |        |     |         |        |         |     | 0.2351 |           |        |     |                        |     | 0.8289 |
| w/oCross-modal                      |     | 93.79% |     | -2.77%  | 7.3051 | -0.5202 |     |        |           |        |     |                        |     |        |
| w/oRepulsionweight                  |     | 63.74% |     | -32.82% | 0.5684 | -7.2569 |     |        |           |        |     |                        |     |        |
|                                     |     |        |     |         |        |         |     |        |           | 0.4063 |     | “appears to be absent” |     |        |
| FullModel                           |     | 96.56% |     | –       | 7.8253 |         | –   | 0.0828 |           |        |     |                        |     | 0.4063 |
Table3:AblationonCC-NegevaluatingtheLensandFrame 0.3862 “may not be”
|          |       |           |       |     |          |     |       | 0.0705 |     |     |     |     |     | 0.6475 |
| -------- | ----- | --------- | ----- | --- | -------- | --- | ----- | ------ | --- | --- | --- | --- | --- | ------ |
| modules. | “Acc” | and “FAR” | refer | to  | Accuracy | and | false |        |     |     |     |     |     |        |
alignmentraterespectively,∆denotestheperformancedrop 0.0 0.2 0.4 0.6 0.8
relativetothefullmodel.
Figure4:Distributionofpredictedrepulsionweightλunder
|     |     |     |     |     |     |     |     | varying | negation | strengths. | Stronger | negations |     | (e.g., “no”) |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---------- | -------- | --------- | --- | ------------ |
yieldhigherλ,confirmingthemodel’sabilitytoadaptively
| ple, and | N is the | total number |     | of samples. | Higher | FAR | in- |     |     |     |     |     |     |     |
| -------- | -------- | ------------ | --- | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
modulatesemanticrepulsion.
| dicates        | stronger | discriminative |        | abilitybetween |             | affirmations |     |     |     |     |     |     |     |     |
| -------------- | -------- | -------------- | ------ | -------------- | ----------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| and negations. | The      | results        | of the | ablation       | experiments |              | are |     |     |     |     |     |     |     |
showninTable3.
|     |     |     |     |     |     |     |     | (32.82% | accuracy | decrease) | underscores |     | its critical | impor- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | --------- | ----------- | --- | ------------ | ------ |
EffectofSyntacticStream.Removingthesyntacticstream tance. This ablation result motivates a deeper investigation
resultsina2.47%decreaseinaccuracyandamoresubstan- intotheunderlyingmechanismsdrivingitscontribution.We
tial3.33-pointdeclineinFAR,demonstratingitscriticalrole
hypothesizethatthemodule’sprimaryfunctionistodynam-
| in enhancing | the | model’s | structural | sensitivity |     | to negation. |     |     |     |     |     |     |     |     |
| ------------ | --- | ------- | ---------- | ----------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
icallycalibraterepulsionstrengthaccordingtothelinguistic
Inparticular,thereductioninFARindicatesdiminishedcon-
intensityofnegation.Totestthishypothesis,weevaluatethe
fidenceinnegationdetection,underscoringtheessentialrole predictedλvaluesacrossacontrolledspectrumofnegation
ofsyntacticgroundinginachievingreliablealignment. intensities.UsingQwen2.5-72B(Yangetal.2024),wegen-
EffectofSemanticStream.Excludingthesemanticstream erate four recaptioned variants for 500 randomly sampled
| results in | a 1.63% | accuracy | drop | and | a 1.73-point |     | reduc- |        |           |      |               |     |         |               |
| ---------- | ------- | -------- | ---- | --- | ------------ | --- | ------ | ------ | --------- | ---- | ------------- | --- | ------- | ------------- |
|            |         |          |      |     |              |     |        | CC-Neg | examples, | each | corresponding |     | to four | distinct lev- |
tioninFAR.Theseresultsindicatethatwhilesyntacticpat- elsofnegationstrength:strong(“no”),moderate(“notany”),
ternseffectivelylocatenegationcues,accurateinterpretation weak(“appearstobeabsent”),andweakest(“maynotbe”).
requires access to global semantic context. The semantic AsshowninFigure4,thepredictedλdistributionsexhibit
stream,derivedfromthefinalCLIPlayer,enablesthemodel
|     |     |     |     |     |     |     |     | a consistent | decreasing |     | trend aligned | with | the | reduction in |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | --- | ------------- | ---- | --- | ------------ |
toidentifythetargetofnegationbycapturingsentence-level
|     |     |     |     |     |     |     |     | negation | strength. | These | results | demonstrate | that | the mod- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ----- | ------- | ----------- | ---- | -------- |
meaning. Without this semantic grounding, the model ex- ule adaptively modulates repulsion intensity in response to
| hibitsreducedperformanceindisambiguatingnegation,par- |     |     |     |     |     |     |     | linguisticcues. |     |     |     |     |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- |
ticularlywhenpolaritylacksexplicitsyntacticmarkers.
| Effect of | Residual | Gating. | Disabling |     | the residual |     | gating |     |     |     |     |     |     |     |
| --------- | -------- | ------- | --------- | --- | ------------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
Conclusion
| leads to        | a substantial  | 27.63%     | drop           | in           | accuracy    | and         | a 7.04- |           |           |               |              |            |             |             |
| --------------- | -------------- | ---------- | -------------- | ------------ | ----------- | ----------- | ------- | --------- | --------- | ------------- | ------------ | ---------- | ----------- | ----------- |
|                 |                |            |                |              |             |             |         | In this   | paper,    | we present    | CLIPGLASSES, |            | a novel     | frame-      |
| point reduction |                | in FAR,    | demonstrating  |              | severe      | degradation |         |           |           |               |              |            |             |             |
|                 |                |            |                |              |             |             |         | work that | addresses | CLIP’s        | limitations  |            | in negation | under-      |
| in both         | classification | and        | discriminative |              | confidence. |             | This    |           |           |               |              |            |             |             |
|                 |                |            |                |              |             |             |         | standing. | Our       | non-intrusive | design       | introduces |             | cognitively |
| validates       | the critical   | importance |                | of balancing |             | syntactic   | at-     |           |           |               |              |            |             |             |
tention with the original semantic representation. Without inspired modules: a Lens for disentangling negated seman-
|     |     |     |     |     |     |     |     | tics and | a Frame | for modeling |     | context-aware | repulsion, | to- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ------------ | --- | ------------- | ---------- | --- |
residualgating,themodelbecomessusceptibletooverfitting
|                 |                   |     |              |     |            |           |      | gether enabling |           | negation-sensitive |            | similarity |             | computation |
| --------------- | ----------------- | --- | ------------ | --- | ---------- | --------- | ---- | --------------- | --------- | ------------------ | ---------- | ---------- | ----------- | ----------- |
| on structurally | salient           | yet | semantically |     | irrelevant | patterns, |      |                 |           |                    |            |            |             |             |
|                 |                   |     |              |     |            |           |      | without         | modifying | CLIP’s             | pretrained |            | parameters. | Experi-     |
| particularly    | in linguistically |     | ambiguous    |     | contexts.  | The       | gat- |                 |           |                    |            |            |             |             |
mentsshowthatourmethodachievescompetitivein-domain
ingmechanismenablesthemodeltopreservecoresentence
meaningwhenstructuralalignmentisuncertain,therebyen- accuracy, state-of-the-art cross-domain generalization and
|     |     |     |     |     |     |     |     | low-resource |     | robustness, | all while | preserving |     | CLIP’s na- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----------- | --------- | ---------- | --- | ---------- |
hancingtherobustnessandstabilityofnegationreasoning.
tivezero-shotcapabilities.Nevertheless,asharedlimitation
EffectofCross-modalContext.Disablingcross-modalin-
acrosscurrentmethodsremainsinhandlingnon-visualnega-
| teraction       | results | in a 2.77% | accuracy |      | decline        | and | a 0.52- |              |      |              |        |      |              |       |
| --------------- | ------- | ---------- | -------- | ---- | -------------- | --- | ------- | ------------ | ---- | ------------ | ------ | ---- | ------------ | ----- |
|                 |         |            |          |      |                |     |         | tions (e.g., | “not | authentic”). | Future | work | will explore | inte- |
| point reduction |         | in FAR.    | Under    | this | configuration, |     | the re- |              |      |              |        |      |              |       |
gratingcommonsenseknowledgetoaddresssuchcases.
| pulsion | weight   | λ is estimated |     | solely  | from textual | features, |     |     |     |     |     |     |     |     |
| ------- | -------- | -------------- | --- | ------- | ------------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
| thereby | limiting | the model’s    |     | ability | to adjust    | repulsion |     |     |     |     |     |     |     |     |
Acknowledgements
| strength | based on | visual | content–a | critical | component |     | for |     |     |     |     |     |     |     |
| -------- | -------- | ------ | --------- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
context-awarenegationunderstanding. This work was supported by the National Natural Science
EffectofDynamicRepulsionWeight.Weobservesubstan- Foundation of China (General Program, No. 62377024,
| tial performance |     | degradation | upon | removing |     | this | module | 2024–2027). |     |     |     |     |     |     |
| ---------------- | --- | ----------- | ---- | -------- | --- | ---- | ------ | ----------- | --- | --- | --- | --- | --- | --- |
10984

References better fine-grained understanding. Advances in Neural In-
formationProcessingSystems,37:27896–27918.
| Alhamoud, | K.; | Alshammari, |     | S.; Tian, | Y.; | Li, | G.; Torr, |     |     |     |     |     |     |     |     |
| --------- | --- | ----------- | --- | --------- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
P.; Kim, Y.; and Ghassemi, M. 2025. Vision-language Kaup, B.; Zwaan, R. A.; and Lu¨dtke, J. 2007. The expe-
| models | do not | understand |     | negation. |     | arXiv | preprint |          |         |          |                |     |     |     |          |
| ------ | ------ | ---------- | --- | --------- | --- | ----- | -------- | -------- | ------- | -------- | -------------- | --- | --- | --- | -------- |
|        |        |            |     |           |     |       |          | riential | view of | language | comprehension: |     | How | is  | negation |
arXiv:2501.09425. represented. Higher level language processes in the brain:
Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer nor- Inferenceandcomprehensionprocesses,255–288.
| malization. | arXivpreprintarXiv:1607.06450. |     |     |     |     |     |     |          |          |      |        |          |       |       |      |
| ----------- | ------------------------------ | --- | --- | --- | --- | --- | --- | -------- | -------- | ---- | ------ | -------- | ----- | ----- | ---- |
|             |                                |     |     |     |     |     |     | Kim, T.; | Lee, S.; | Kim, | S.-W.; | and Kim, | D.-J. | 2025. | Vip- |
Caffagni, D.; Sarto, S.; Cornia, M.; Baraldi, L.; and cap:Retrievaltext-basedvisualpromptsforlightweightim-
Cucchiara, R. 2025. Recurrence-Enhanced Vision-and- agecaptioning. InProceedingsoftheAAAIConferenceon
ArtificialIntelligence(AAAI).
| Language | Transformers |     | for Robust |     | Multimodal | Document |     |     |     |     |     |     |     |     |     |
| -------- | ------------ | --- | ---------- | --- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Retrieval. In Proceedings of the IEEE/CVF Conference on Ko,H.;andPark,C.-M.2025. BringingCLIPtotheClinic:
ComputerVisionandPatternRecognition(CVPR). Dynamic Soft Labels and Negation-Aware Learning for
Cai, T. T.; and Ma, R. 2022. Theoretical Foundations of MedicalAnalysis. InProceedingsoftheIEEE/CVFConfer-
t-SNE for Visualizing High-Dimensional Clustered Data. enceonComputerVisionandPatternRecognition(CVPR),
| JournalofMachineLearningResearch,23(301):1–54. |     |     |     |     |     |     |     | 25897–25906. |     |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei- Koishigarina,D.;Uselis,A.;andOh,S.J.2025. CLIPBe-
Fei, L. 2009. Imagenet: A large-scale hierarchical image haves like a Bag-of-Words Model Cross-modally but not
database. In Proceedings of the IEEE/CVF Conference on Uni-modally. arXivpreprintarXiv:2502.03566.
ComputerVisionandPatternRecognition(CVPR).
|     |     |     |     |     |     |     |     | Li, B.; Dong, | H.; | Zhang, | D.; | Zhao, Z.; | Gao, | J.; and | Li, X. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------ | --- | --------- | ---- | ------- | ------ |
Fang, S.; Wang, Y.; Zhang, J.; Li, Z.; and Wang, Y. 2025. 2025a. ExploringEfficientOpen-VocabularySegmentation
| Adaptive | Language-Aware |          | Image       | Reflection |        | Removal       | Net- |                     |     |     |                                |     |     |     |     |
| -------- | -------------- | -------- | ----------- | ---------- | ------ | ------------- | ---- | ------------------- | --- | --- | ------------------------------ | --- | --- | --- | --- |
|          |                |          |             |            |        |               |      | intheRemoteSensing. |     |     | arXivpreprintarXiv:2509.12040. |     |     |     |     |
| work.    | In Kwok,       | J., ed., | Proceedings |            | of the | Thirty-Fourth |      |                     |     |     |                                |     |     |     |     |
International Joint Conference on Artificial Intelligence, Li, B.; Zhang, D.; Zhao, Z.; Gao, J.; and Li, X. 2025b.
Stitchfusion:Weavinganyvisualmodalitiestoenhancemul-
IJCAI-25,973–981.InternationalJointConferencesonAr-
|                                   |     |     |     |            |     |     |     | timodalsemanticsegmentation. |     |     |     | InProceedingsofthe33rd |     |     |     |
| --------------------------------- | --- | --- | --- | ---------- | --- | --- | --- | ---------------------------- | --- | --- | --- | ---------------------- | --- | --- | --- |
| tificialIntelligenceOrganization. |     |     |     | MainTrack. |     |     |     |                              |     |     |     |                        |     |     |     |
ACMInternationalConferenceonMultimedia,1308–1317.
Fang,X.;Liu,D.;Fang,W.;Zhou,P.;Xu,Z.;Xu,W.;Chen,
|             |     |             |        |        |              |     |       | Li, F.-F.; | Andreeto, | M.; | Ranzato, | M.; | and Perona, |     | P. 2022. |
| ----------- | --- | ----------- | ------ | ------ | ------------ | --- | ----- | ---------- | --------- | --- | -------- | --- | ----------- | --- | -------- |
| J.; and Li, | R.  | 2024. Fewer | steps, | better | performance: |     | Effi- |            |           |     |          |     |             |     |          |
Caltech101.
cientcross-modalcliptrimmingforvideomomentretrieval
usinglanguage. InProceedingsoftheAAAIConferenceon Li, H.; Ding, L.; Fang, M.; and Tao, D. 2024. Revisit-
ArtificialIntelligence,volume38,1735–1743. ing catastrophic forgetting in large language model tuning.
arXivpreprintarXiv:2406.04836.
Feng,Z.;Guo,Q.;Xiao,X.;Xu,R.;Yang,M.;andZhang,
S.2025. UnifiedVideoGenerationviaNext-SetPrediction Li, K.; Jiang, Z.; Shen, Z.; Wang, Z.; Lv, C.; Zhang, S.;
inContinuousDomain.InProceedingsoftheIEEE/CVFIn-
|     |     |     |     |     |     |     |     | Wu, F.; | and Wu, | F. 2025c. | MadaKV: |     | Adaptive | Modality- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | --------- | ------- | --- | -------- | --------- | --- |
ternationalConferenceonComputerVision,19427–19438. Perception KV Cache Eviction for Efficient Multimodal
Geirhos,R.;Jacobsen,J.-H.;Michaelis,C.;Zemel,R.;Bren- Long-ContextInference. InProceedingsofthe63rdAnnual
|          |         |         |           |        |         |       |          | Meeting | of the  | Association | for | Computational |         | Linguistics |      |
| -------- | ------- | ------- | --------- | ------ | ------- | ----- | -------- | ------- | ------- | ----------- | --- | ------------- | ------- | ----------- | ---- |
| del, W.; | Bethge, | M.; and | Wichmann, |        | F. A.   | 2020. | Shortcut |         |         |             |     |               |         |             |      |
|          |         |         |           |        |         |       |          | (Volume | 1: Long | Papers),    | ACL | 2025,         | Vienna, | Austria,    | July |
| learning | in deep | neural  | networks. | Nature | Machine |       | Intelli- |         |         |             |     |               |         |             |      |
27-August1,2025,13306–13318.AssociationforCompu-
gence,2(11):665–673.
tationalLinguistics.
| Hendrycks,D.;andGimpel,K.2016. |     |     |     |     | Gaussianerrorlinear |     |     |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
units(gelus). arXivpreprintarXiv:1606.08415. Lin, Y.; Chen, M.; Zhang, K.; Li, H.; Li, M.; Yang, Z.; Lv,
Janssens, R.; Wolfert, P.; Demeester, T.; and Belpaeme, T. D.;Lin,B.;Liu,H.;andCai,D.2024. Tagclip:Alocal-to-
|                                     |     |        |         |      |                    |        |     | global framework |            | to   | enhance       | open-vocabulary |     | multi-label |     |
| ----------------------------------- | --- | ------ | ------- | ---- | ------------------ | ------ | --- | ---------------- | ---------- | ---- | ------------- | --------------- | --- | ----------- | --- |
| 2024. Integrating                   |     | visual | context | into | language           | models | for |                  |            |      |               |                 |     |             |     |
|                                     |     |        |         |      |                    |        |     | classification   | of         | clip | without       | training.       | In  | Proceedings | of  |
| situatedsocialconversationstarters. |     |        |         |      | IEEETransactionson |        |     |                  |            |      |               |                 |     |             |     |
|                                     |     |        |         |      |                    |        |     | the AAAI         | Conference |      | on Artificial | Intelligence,   |     | volume      | 38, |
AffectiveComputing,16(1):223–236.
3513–3521.
| Jha, S.; | Gong, | D.; and       | Yao, L.    | 2024. | Clap4clip: | Continual       |     |         |       |          |        |         |     |      |         |
| -------- | ----- | ------------- | ---------- | ----- | ---------- | --------------- | --- | ------- | ----- | -------- | ------ | ------- | --- | ---- | ------- |
|          |       |               |            |       |            |                 |     | Ma, Z.; | Hong, | J.; Gul, | M. O.; | Gandhi, | M.; | Gao, | I.; and |
| learning | with  | probabilistic | finetuning |       | for        | vision-language |     |         |       |          |        |         |     |      |         |
models.Advancesinneuralinformationprocessingsystems, Krishna, R. 2023. Crepe: Can vision-language founda-
37:129146–129186. tion models reason compositionally? In Proceedings of
theIEEE/CVFConferenceonComputerVisionandPattern
Jiang,Z.;Xu,J.;Zhang,S.;Shen,T.;Li,J.;Kuang,K.;Cai,
Recognition(CVPR).
| H.;andWu,F.2025. |     | FedCFA:AlleviatingSimpson’sPara- |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dox in Model Aggregation with Counterfactual Federated Morante,R.;andBlanco,E.2021. Recentadvancesinpro-
|                                                      |                                           |     |     |     |     |     |     | cessing  | negation. | Natural | Language |     | Engineering, |     | 27(2): |
| ---------------------------------------------------- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | --------- | ------- | -------- | --- | ------------ | --- | ------ |
| Learning.                                            | InAAAI-25,SponsoredbytheAssociationforthe |     |     |     |     |     |     |          |           |         |          |     |              |     |        |
| AdvancementofArtificialIntelligence,February25-March |                                           |     |     |     |     |     |     | 121–130. |           |         |          |     |              |     |        |
4,2025,Philadelphia,PA,USA,17662–17670. Nam, J.; Im, J.; Kim, W.; and Kil, T. 2025. Extract free
Jing, D.; He, X.; Luo, Y.; Fei, N.; Wei, W.; Zhao, H.; Lu, densemisalignmentfromCLIP. InProceedingsoftheAAAI
Z.;etal.2024. Fineclip:Self-distilledregion-basedclipfor ConferenceonArtificialIntelligence(AAAI).
10985

Orenes, I.; Beltra´n, D.; and Santamar´ıa, C. 2014. How Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
negation is understood: Evidence from the visual world Li,C.;Liu,D.;Huang,F.;Wei,H.;Lin,H.;Yang,J.;Tu,J.;
paradigm. Journalofmemoryandlanguage,74:36–45. Zhang,J.;Yang,J.;Yang,J.;Zhou,J.;Lin,J.;Dang,K.;Lu,
|           |          |       |         |           |     |           |     | K.; Bao, | K.; Yang, | K.; Yu, | L.; Li, M.; | Xue, | M.; Zhang, | P.; |
| --------- | -------- | ----- | ------- | --------- | --- | --------- | --- | -------- | --------- | ------- | ----------- | ---- | ---------- | --- |
| Park, J.; | Lee, J.; | Song, | J.; Yu, | S.; Jung, | D.; | and Yoon, | S.  |          |           |         |             |      |            |     |
Zhu,Q.;Men,R.;Lin,R.;Li,T.;Xia,T.;Ren,X.;Ren,X.;
| 2025. Know” | No” | Better: | A   | Data-Driven |     | Approach | for |     |     |     |     |     |     |     |
| ----------- | --- | ------- | --- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Fan,Y.;Su,Y.;Zhang,Y.;Wan,Y.;Liu,Y.;Cui,Z.;Zhang,
| Enhancing | Negation | Awareness |     | in CLIP. |     | arXiv preprint |     |         |               |         |           |     |         |       |
| --------- | -------- | --------- | --- | -------- | --- | -------------- | --- | ------- | ------------- | ------- | --------- | --- | ------- | ----- |
|           |          |           |     |          |     |                |     | Z.; and | Qiu, Z. 2024. | Qwen2.5 | Technical |     | Report. | arXiv |
arXiv:2501.10913.
preprintarXiv:2412.15115.
Qin,X.;Zhang,P.;Yang,J.J.O.;Zeng,G.;Li,Y.;Wang,Y.;
Yuksekgonul,M.;Bianchi,F.;Kalluri,P.;Jurafsky,D.;and
| Zhang,W.;andDai,P.2025. |     |     | CLIPisAlmostAllYouNeed: |     |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Zou,J.2022.Whenandwhyvision-languagemodelsbehave
| Towards | Parameter-Efficient |     | Scene | Text | Retrieval | without |     |                                       |     |     |     |     |               |     |
| ------- | ------------------- | --- | ----- | ---- | --------- | ------- | --- | ------------------------------------- | --- | --- | --- | --- | ------------- | --- |
|         |                     |     |       |      |           |         |     | likebags-of-words,andwhattodoaboutit? |     |     |     |     | arXivpreprint |     |
OCR.InProceedingsoftheIEEE/CVFConferenceonCom-
arXiv:2210.01936.
puterVisionandPatternRecognition(CVPR).
Zhai,Y.;Tong,S.;Li,X.;Cai,M.;Qu,Q.;Lee,Y.J.;andMa,
| Quantmeyer, | V.;  | Mosteiro, | P.;     | and Gatt, | A.  | 2024.          | How |                                     |               |     |              |                |     |        |
| ----------- | ---- | --------- | ------- | --------- | --- | -------------- | --- | ----------------------------------- | ------------- | --- | ------------ | -------------- | --- | ------ |
|             |      |           |         |           |     |                |     | Y. 2024.                            | Investigating | the | catastrophic | forgetting     | in  | multi- |
| and where   | does | CLIP      | process | negation? |     | arXiv preprint |     |                                     |               |     |              |                |     |        |
|             |      |           |         |           |     |                |     | modallargelanguagemodelfine-tuning. |               |     |              | InConferenceon |     |        |
arXiv:2407.10488.
ParsimonyandLearning,202–227.PMLR.
Radford,A.;Kim,J.W.;Hallacy,C.;Ramesh,A.;Goh,G.;
|          |             |     |         |              |     |            |     | Zhang,       | S.; Xu, Y.; | Usuyama, | N.; Xu,  | H.; Bagga, | J.;        | Tinn, |
| -------- | ----------- | --- | ------- | ------------ | --- | ---------- | --- | ------------ | ----------- | -------- | -------- | ---------- | ---------- | ----- |
| Agarwal, | S.; Sastry, | G.; | Askell, | A.; Mishkin, |     | P.; Clark, | J.; |              |             |          |          |            |            |       |
|          |             |     |         |              |     |            |     | R.; Preston, | S.; Rao,    | R.;      | Wei, M.; | Valluri,   | N.; et al. | 2023. |
et al. 2021. Learning transferable visual models from nat- Biomedclip: a multimodal biomedical foundation model
ural language supervision. In International conference on pretrained from fifteen million scientific image-text pairs.
MachineLearning(ICML).
arXivpreprintarXiv:2303.00915.
Rombach,R.;Blattmann,A.;Lorenz,D.;Esser,P.;andOm- Zhang,T.;Liu,P.;Lu,Y.;Cai,M.;Zhang,Z.;Zhang,Z.;and
mer,B.2022. High-ResolutionImageSynthesiswithLatent Zhou, Q. 2025a. Cwnet: Causal wavelet network for low-
DiffusionModels.InProceedingsoftheIEEE/CVFConfer-
|     |     |     |     |     |     |     |     | lightimageenhancement. |     |     | InProceedingsoftheIEEE/CVF |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | -------------------------- | --- | --- | --- |
enceonComputerVisionandPatternRecognition(CVPR).
InternationalConferenceonComputerVision,8789–8799.
Singh,J.;Shrivastava,I.;Vatsa,M.;Singh,R.;andBharati,
Zhang,T.;Liu,P.;Zhang,Z.;andZhou,Q.2025b.SPJFNet:
A.2025. LearningthePowerof”No”:FoundationModels Self-MiningPrior-GuidedJointFrequencyEnhancementfor
withNegations.InProceedingsoftheWinterConferenceon Ultra-Efficient Dark Image Restoration. arXiv preprint
| ApplicationsofComputerVision(WACV). |     |     |     |     |     |     |     | arXiv:2508.04041. |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
Su, X.; Mao, Q.; Wu, Z.; Lin, X.; You, S.; Liao, Y.; and Zhang, T.; Liu, P.; Zhong, Z.; Zhang, Z.; and Zhou, Q.
Xu,C.2025. Largelanguagemodelsdrivenneuralarchitec- 2025c. Beyond Illumination: Fine-Grained Detail Preser-
ture search for universal and lightweight disease diagnosis vation in Extreme Dark Image Restoration. arXiv preprint
arXiv:2508.03336.
| onhistopathologyslideimages. |     |     |     | npjDigitalMedicine,8(1): |     |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
682. Zhang, X.; Li, J.; Zhang, J.; Dang, Z.; Ren, J.; Bo, L.; and
|          |              |        |          |                   |           |     |       | Tu,Z.2025d.                           | SemTalk:HolisticCo-speechMotionGener- |     |     |     |               |     |
| -------- | ------------ | ------ | -------- | ----------------- | --------- | --- | ----- | ------------------------------------- | ------------------------------------- | --- | --- | --- | ------------- | --- |
| Sun, L.; | Wang, J.;    | Zhang, | K.; Su,  | Y.;               | and Weng, | F.  | 2021. |                                       |                                       |     |     |     |               |     |
|          |              |        |          |                   |           |     |       | ationwithFrame-levelSemanticEmphasis. |                                       |     |     |     | InProceedings |     |
| RpBERT:  | a text-image |        | relation | propagation-based |           |     | BERT  |                                       |                                       |     |     |     |               |     |
model for multimodal NER. In Proceedings of the AAAI oftheIEEE/CVFInternationalConferenceonComputerVi-
conference on artificial intelligence, volume 35, 13860– sion,13761–13771.
13868.
|     |     |     |     |     |     |     |     | Zhang, | X.; Li, J.; | Zhang, | J.; Ren, | J.; Bo, | L.; and | Tu, Z. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ----------- | ------ | -------- | ------- | ------- | ------ |
Sun, Z.; Fang, Y.; Wu, T.; Zhang, P.; Zang, Y.; Kong, S.; 2025e. EchoMask: Speech-Queried Attention-based Mask
|                |          |          |          |       |                |     |        | Modeling    | for Holistic | Co-Speech |               | Motion Generation. |            | In  |
| -------------- | -------- | -------- | -------- | ----- | -------------- | --- | ------ | ----------- | ------------ | --------- | ------------- | ------------------ | ---------- | --- |
| Xiong, Y.;     | Lin, D.; | and      | Wang, J. | 2024. | Alpha-clip:    |     | A clip |             |              |           |               |                    |            |     |
|                |          |          |          |       |                |     |        | Proceedings | of the       | 33rd ACM  | International |                    | Conference | on  |
| model focusing | on       | wherever | you      | want. | In Proceedings |     | of     |             |              |           |               |                    |            |     |
Multimedia,10827–10836.
| the IEEE/CVF | conference |     | on computer |     | vision | and pattern |     |     |     |     |     |     |     |     |
| ------------ | ---------- | --- | ----------- | --- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
recognition,13019–13029. Zuanazzi,A.;Ripolle´s,P.;Lin,W.M.;Gwilliams,L.;King,
|     |     |     |     |     |     |     |     | J.-R.;andPoeppel,D.2024. |     |     | Negationmitigatesratherthan |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --------------------------- | --- | --- | --- |
Thrush,T.;Jiang,R.;Bartolo,M.;Singh,A.;Williams,A.;
|                          |     |     |                          |     |     |     |     | inverts the | neural representations |     | of  | adjectives. | PLoS | biol- |
| ------------------------ | --- | --- | ------------------------ | --- | --- | --- | --- | ----------- | ---------------------- | --- | --- | ----------- | ---- | ----- |
| Kiela,D.;andRoss,C.2022. |     |     | Winoground:Probingvision |     |     |     |     |             |                        |     |     |             |      |       |
ogy,22(5):e3002622.
| and language   | models | for          | visio-linguistic |            | compositionality. |             |     |     |     |     |     |     |     |     |
| -------------- | ------ | ------------ | ---------------- | ---------- | ----------------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
| In Proceedings | of     | the IEEE/CVF |                  | Conference |                   | on Computer |     |     |     |     |     |     |     |     |
VisionandPatternRecognition(CVPR).
| Vaswani,                                      | A.; Shazeer, | N.;                             | Parmar, | N.; | Uszkoreit, | J.; | Jones, |     |     |     |     |     |     |     |
| --------------------------------------------- | ------------ | ------------------------------- | ------- | --- | ---------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
| L.;Gomez,A.N.;Kaiser,Ł.;andPolosukhin,I.2017. |              |                                 |         |     |            |     | At-    |     |     |     |     |     |     |     |
| tentionisallyouneed.                          |              | Advancesinneuralinformationpro- |         |     |            |     |        |     |     |     |     |     |     |     |
cessingsystems,30.
| Xing, F.;                              | Li, M.; | Wang, | Y.-G.; Zhu, | G.; | and | Cao, X.      | 2024. |     |     |     |     |     |     |     |
| -------------------------------------- | ------- | ----- | ----------- | --- | --- | ------------ | ----- | --- | --- | --- | --- | --- | --- | --- |
| Clipvqa:Videoqualityassessmentviaclip. |         |       |             |     |     | IEEETransac- |       |     |     |     |     |     |     |     |
tionsonBroadcasting.
10986
