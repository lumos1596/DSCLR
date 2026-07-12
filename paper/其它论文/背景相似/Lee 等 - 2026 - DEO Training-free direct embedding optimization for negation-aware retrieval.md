DEO: Training-Free Direct Embedding Optimization for
Negation-Aware Retrieval
TaegyeongLee1* JiwonPark2* SeunghyunHwang3* JooYoungJang1†
1Miri.DIH 2DepartmentofIndustrialEngineering,HanyangUniversity
3DepartmentofAppliedDataScience,SungkyunkwanUniversity
tglee@miridih.com, jiwonpark@hanyang.ac.kr,
hsh1030@g.skku.edu, jyjang@miridih.com
Abstract to improve retrieval performance through fine-
tuningembeddingmodels. Whileeffective,these
Recent advances in Large Language Models approaches require substantial GPU resources,
(LLMs)andRetrieval-AugmentedGeneration
large-scalefine-tuningdatasets,andextensivetrain-
(RAG) have enabled diverse retrieval meth-
ing, which restrict their applicability in resource-
ods. However, existing retrieval methods of-
constrainedenvironments. Inaddition,suchmeth-
tenfailtoaccuratelyretrieveresultsfornega-
odscanpotentiallydegraderetrievalperformance
tion and exclusion queries. To address this
limitation,priorapproachesrelyonembedding and lack clear controllability, particularly in han-
adaptationorfine-tuning,whichintroduceaddi- dling queries involving negation and exclusion.
tionalcomputationalcostanddeploymentcom- Consequently,effectivelycapturinguserintentin
plexity. We propose Direct Embedding Op- retrievalunderquerieswithnegationandexclusion
timization(DEO),atraining-freemethodfor
remains both challenging and important (Singh
negation-awaretextandmultimodalretrieval.
etal.,2023;Alhamoudetal.,2025).
DEOdecomposesqueriesintopositiveandneg-
Recently, a method has leveraged sparse
ativecomponentsandoptimizesthequeryem-
bedding with a contrastive objective. With- autoencoders (SAE) to interpret and control
outadditionaltrainingdataormodelupdates, dense embeddings through latent sparse features,
DEOoutperformsbaselinesonNegConstraint, while maintaining comparable retrieval accuracy.
withgainsof+0.0738nDCG@10and+0.1028 NUDGE (Zeighami et al., 2024), on the other
MAP@100, while improving Recall@5 by
hand,proposesanon-parametricembeddingfine-
+6%overOpenAICLIPinmultimodalretrieval.
tuningapproachthatdirectlymodifiesembeddings
These results demonstrate the practicality of
of data records to maximize k-NNretrieval accu-
DEO for negation- and exclusion-aware re-
racy, achieving significant improvements in both
trievalinreal-worldsettings. Thecodeispub-
liclyavailableatGitHub. performance and efficiency over full model fine-
tuningandadaptor-basedmethods. However,these
1 Introduction methodsstillrelyonfine-tuningandthusrequire
largedatasetsandsubstantialGPUresources,lim-
Recent advances in Large Language Models
itingtheirpracticality.
(LLMs)(Grattafiorietal.,2024;Yangetal.,2025;
Therefore,weproposeDirectEmbeddingOpti-
Teametal.,2024)andRetrieval-AugmentedGen-
mization(DEO),asimpleyeteffectiveapproach
eration(RAG)(Lewisetal.,2020;Xuetal.,2025;
formultimodalretrievalthatdoesnotrequirefine-
Zeighami et al., 2024) have enabled systems to
tuning. First,DEOdecomposesuserqueriesinto
generateresponsesconditionedonuserinputsand
positiveandnegativesub-queriesusingLLMs. For
aligned with user intent. However, real-world
example,givenaquerysuchas“showmethelatest
queriesfrequentlyincludenegationandexclusion,
earnings forecast, but exclude 2024 results,” the
posingchallengesforconsistentlyretrievingcon-
LLM generates positive queries (e.g., “earnings
tentthataccuratelyreflectsuserintent(Caietal.,
forecastfor2025,”“financialstatements”)andneg-
2025;Singhetal.,2023;Dongetal.,2023).
ativequeries(e.g.,“2024earnings,”“2024financial
Existing approaches (Zeighami et al., 2024;
report”), thereby explicitly separating user intent
Shevkunovetal.;Wangetal.,2024)haveattempted
intoinclusionandexclusioncomponents.
Second,boththeoriginaluserqueryandthede-
*Equalcontribution
†Correspondingauthor composedpositiveandnegativesub-queriesareem-
1
6202
raM
01
]LC.sc[
1v58190.3062:viXra

beddedusingapre-trainedembeddingmodel. We et al., 2020). Prior research has explored im-
directlyoptimizetheuserqueryembeddingspace provementsthroughtrainingstrategies,distillation,
viaacontrastiveloss,pullingitclosertotheposi- andpre-training. Transferlearningonlarge-scale
tivequeryembeddingsandpushingitawayfrom datasetssuchasMSMARCO(Singhetal.,2023)
thenegativequeryembeddings. Thisoptimization hasalsobeenwidelyadopted,thoughitisresource-
alignstheoriginalqueryembeddingwithmorespe- intensivetoconstruct(Bajajetal.,2016). Morere-
cificandsemanticallyrichpositivequeries,while cently,zero-shotdenseretrievalhasreducedthere-
simultaneouslydistancingitfromnegativequeries, lianceonexplicitrelevancelabels. Despitethesead-
therebyproducingembeddingsthatbetterrepresent vances,theproblemofeffectivelyhandlingqueries
userintentinvolvingnegationandexclusion. withnegationandexclusionremainsrelativelyun-
| Finally, retrieval  | is performed |          | using the | op- | derexplored. |     |     |     |     |     |
| ------------------- | ------------ | -------- | --------- | --- | ------------ | --- | --- | --- | --- | --- |
| timized embeddings, | enabling     | improved | perfor-   |     |              |     |     |     |     |     |
mancewithoutadditionalfine-tuning.
|     |     |     |     |     | 2.2 EmbeddingControlandFine-tuning |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- |
OurexperimentsshowthatDEOimprovesMAP
Alternatives
| on the NegConstraint | benchmark | from | 0.6299 | to  |     |     |     |     |     |     |
| -------------------- | --------- | ---- | ------ | --- | --- | --- | --- | --- | --- | --- |
0.7327 and nDCG@10 from 0.7139 to 0.7877 Beyondfullmodelfine-tuning,recentworkhasex-
in the best-performing configuration, using the ploredmethodsfordirectlycontrollingorrefining
BGE-large-en-v1.5 embedding model. On mul- embeddingspacetoimproveretrieval. Represen-
tativeapproachesincludeprojectingdenseembed-
| timodal retrieval | (COCO-Neg) | (Alhamoud |     | et al., |     |     |     |     |     |     |
| ----------------- | ---------- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- |
2025),DEOnotablyincreasesRecall@5withOpe- dingsintointerpretablesparselatentfeaturesand
nAI CLIP (Radford et al., 2021) from 0.4792 to applying non-parametric optimization to directly
0.5392. Moreover,DEOconsistentlyimprovesper- adjustrecordembeddingsforimprovedk-NNac-
formanceacrossallevaluatedtextretrievalbench- curacy (Zeighami et al., 2024; Shevkunov et al.;
marks. Theseresultsdemonstratethatourmethod Wang et al., 2024). While these methods can be
providesstableabsolutegains,makingitrobustfor effective, they often require sizable datasets and
real-world retrieval settings and effective in han- substantialGPUresources,whichlimitstheirappli-
dlingnegationandexclusionqueries. cabilityinresource-constrainedsettings. Thismo-
Ourmaincontributionsareasfollows: tivatesthedevelopmentoflightweightapproaches
|     |     |     |     |     | that better | capture | nuanced |     | aspects | of user in- |
| --- | --- | --- | --- | --- | ----------- | ------- | ------- | --- | ------- | ----------- |
• WeproposeDirectEmbeddingOptimization tent—particularlynegationandexclusionwithout
| (DEO),aneffectiveretrievalmethodwithout |     |     |     |     | additionalfine-tuning. |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- |
fine-tuningoradditionaldatasets.
|     |     |     |     |     | 2.3 Negation-andExclusion-awareRetrieval |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- |
• Bydirectlyoptimizingtheembeddingspace
| via contrastive | loss over | positive | and | nega- |         |            |          |     |              |         |
| --------------- | --------- | -------- | --- | ----- | ------- | ---------- | -------- | --- | ------------ | ------- |
|                 |           |          |     |       | Queries | containing | negation |     | or exclusion | require |
tivesub-queries,DEOenablesnegation-and
anexplicitdistinctionbetweeninclusionandexclu-
exclusion-awareretrievalthatmoreprecisely
sionsemantics,whichstandardretrieversoftenfail
capturesuserintent.
tocapturereliably(Singhetal.,2023;Alhamoud
• DEO is model- and modality-agnostic, gen- etal.,2025). Priorworkhasattemptedtoaddress
thisissuethroughfine-tuningortask-specificregu-
| eralizing | across diverse | embedding | models |     |     |     |     |     |     |     |
| --------- | -------------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
larizationtoimprovesensitivitytonegation(Wang
andretrievalsettings,andexperimentsdemon-
|     |     |     |     |     | et al., 2024; | Zeighami |     | et al., | 2024), | but such ap- |
| --- | --- | --- | --- | --- | ------------- | -------- | --- | ------- | ------ | ------------ |
strateconsistentimprovementsoverbaselines
proachestypicallyincursubstantialcomputational
onbothtextandmultimodalbenchmarks.
|               |     |     |     |     | cost and                                     | offer limited |     | controllability.  |     | An alterna- |
| ------------- | --- | --- | --- | --- | -------------------------------------------- | ------------- | --- | ----------------- | --- | ----------- |
| 2 RelatedWork |     |     |     |     | tivedirectionistoleveragelargelanguagemodels |               |     |                   |     |             |
|               |     |     |     |     | to decompose                                 | queries       |     | into semantically |     | coherent    |
2.1 DenseRetrieval
|     |     |     |     |     | sub-queries, | thereby | providing |     | explicit | represen- |
| --- | --- | --- | --- | --- | ------------ | ------- | --------- | --- | -------- | --------- |
Denseretrievalencodesqueriesanddocumentsinto tations of user intent. Building on this idea, our
continuous,low-dimensionalembeddingsthatcap- methoddirectlyoptimizesqueryembeddingswith
turesemanticsimilarity,andtypicallyoutperforms respecttopositiveandnegativesub-queriesusing
sparse term-matching methods based on high- acontrastiveobjective,whichimprovesalignment
dimensionalvectors(Kangetal.,2025;Karpukhin withuserintentwhileavoidingmodelfine-tuning.
2

Positive query
specific examples of photomontage artworks
biographical details of artists involved in Embedding vector
photomontage
Decompose
specific events or exhibitions featuring
photomontage
any mention of Bayreuth's identity or
geographic location Updated
W of h t a h t e a c r u e l t t u h r e a c l h c a e r n a t c e t r e B ri a s y ti r c e s u a th n d (e i x n c f l l u u d en in c g e s its positive loss embedding vector
Negative Sub-Queries
identity as Bayreuth) and the art form
Photomontage (excluding examples of negative loss
photomontage)? cultural significance and role of Bayreuth as
a cultural hub
Input Query (with negation) historical and social influences of Bayreuth
on regional culture
architectural and infrastructural features of Input query
Bayreuth's cultural institutions Embedding vector Negative query
Embedding vector
Positive Sub-Queries
(a) Input Query Decompose with LLMs (b) Direct Embedding Optimization
Figure1:OverviewoftheproposedDirectEmbeddingOptimization(DEO).(a)Givenaninputquerycontaining
negation,weuseanLLMtodecomposeitintopositiveandnegativesub-queries. (b)Theinputqueryembeddingis
thenoptimizedwithacontrastivelossbypullingitclosertopositivequeryembeddingsandpushingitfartherfrom
negativequeryembeddings,enablingnegation-andexclusion-awareretrieval.
2.4 NegationRobustnessinMultimodal thisapproach,weenablenegation-andexclusion-
Retrieval aware retrieval in the embedding space without
fine-tuningembeddingmodels.
Vision–languageretrieverstrainedwithlarge-scale
contrastivelearning,suchasCLIP(Radfordetal.,
3.1 QueryDecomposition
2021),learnasharedimage–textembeddingspace
andachievestrongzero-shotretrievalperformance. Inreal-worldscenarios,userqueriesfrequentlycon-
BLIP (Li et al., 2022) and BLIP-2 (Li et al., tainnegationorexclusion,expressedusingphrases
2023)extendthesecapabilitiestocaptioningand suchas“exclude”or“donotmention.”Toaddress
VQA(Visual-Question Answering) (Antol et al., thischallenge,asshowninFigure1(a),weemploy
2015;Kimetal.,2021)whilemaintainingcompeti- alargelanguagemodel(LLM)inaprompt-based
tiveretrievalaccuracy. However,despitestrongav- settingtosemanticallyanalyzetheinputqueryand
erageperformance,thesemodelsremainbrittleto explicitlycaptureitsnegationorexclusionintent.
negationphenomena,includingattributenegation TheLLMthendecomposestheoriginalqueryinto
(e.g.,“notred”),absence(e.g.,“noperson”),and structuredpositiveandnegativesub-queries.
relationalnegation(e.g.,“AisnotleftofB”). Neg- For example, given the query “What are the
Bench(Alhamoudetal.,2025)evaluatesthislimi- characteristics and influences of the cultural cen-
tationthroughretrievalandmultiple-choicetasks ter Bayreuth (excluding its identity as Bayreuth)
withnegatedcaptionsacrossimageandvideodo- andtheartformPhotomontage(excludingexam-
mains,confirmingthatexistingmodelsstruggleto plesofphotomontage)?”,ourmethodseparatesthe
distinguishaffirmativefromnegatedstatements. retrieval-relevantcomponentsfromtheexclusion
Incontrast,ourapproachenablesnegation-and constraints and generates corresponding positive
exclusion-awareretrievalacrossbothtextandmul- and negative sub-queries. These sub-queries are
timodalsettingsbyprovidingexplicitcontrolover subsequently embedded and optimized indepen-
semanticcomponentsrelevanttotruthconditions, dently, yielding a structured representation that
anddemonstratesrobustnessunderestablishedeval- more accurately reflects the user’s intent under
uationbenchmarks. negationandexclusion.
The positive sub-queries are an enriched ver-
3 Method
sion of the user’s request: [“cultural significance
Weproposeasimpleyeteffectiveretrievalmethod. androleofBayreuthasaculturalhub","historical
As illustrated in Figure 1, our model consists of andsocialinfluencesofBayreuthonregionalcul-
two stages: (a) Decomposing the user query into ture","architecturalandinfrastructuralfeaturesof
positiveandnegativesub-queries. (b)Directlyop- Bayreuth’sculturalinstitutions”]. Thisexpansion
timizing the embedding space of input query as capturestheessenceoftheuser’sintentwhileex-
a parameter by using contrastive loss. Through pressingitinamoreelaborateandcomprehensive
3

| form. |     |     |     |     |     | Here, | λ p , λ n | , and λ | o are | hyperparameters |     | con- |
| ----- | --- | --- | --- | --- | --- | ----- | --------- | ------- | ----- | --------------- | --- | ---- |
Thenegativesub-queriesexplicitlyencodethe trollingthestrengthofpositiveattraction,negative
exclusionary intent: ["specific examples of pho- repulsion,andconsistencyregularization,respec-
| tomontage | artworks"," | biographical |     | details | of  | tively. |     |     |     |     |     |     |
| --------- | ----------- | ------------ | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- |
artistsinvolvedinphotomontage","specificevents WeminimizeLusingagradient-basedoptimizer
orexhibitionsfeaturingphotomontage","anymen- (Adam)forafixednumberofstepswhilekeeping
tionofBayreuth’sidentityorgeographiclocation"]. theencoderparametersunchanged. Theresulting
Thenegationissemanticallyexpandedsothatthe optimizedembeddinge u isthenusedasthefinal
system can clearly identify and filter out the un- queryrepresentationforretrieval.
wantedelement.
|     |     |     |     |     |     | 3.3 RetrievalwithOptimizedEmbeddings |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- |
Throughthisstructureddecomposition,theorig-
inalquerybecomesbothclarifiedandoperational- Wederiveanoptimizedembeddingspacethatboth
ized: thepositivesub-queriesguidesemanticrele-
enrichestheuserqueryandincorporatesitsnega-
vanceexpansion,whereasthenegativesub-queries tionorexclusionintent. Retrievalisthenperformed
impose explicit exclusion constraints during re- directly with this embedding. To further empha-
| trieval. |     |     |     |     |     | size exclusion |          | or restrict | specific    | documents, |           | the |
| -------- | --- | --- | --- | --- | --- | -------------- | -------- | ----------- | ----------- | ---------- | --------- | --- |
|          |     |     |     |     |     | retrieval      | behavior | can         | be adjusted |            | by tuning | the |
3.2 DirectEmbeddingOptimization
|     |     |     |     |     |     | optimization | hyperparameters |     |     | λ and | λ   | , which |
| --- | --- | --- | --- | --- | --- | ------------ | --------------- | --- | --- | ----- | --- | ------- |
|     |     |     |     |     |     |              |                 |     |     | p     | n   |         |
Unlike previous methods (Zeighami et al., 2024; weight the contributions of positive and negative
|          |                 |         |            |         |     | sub-queries. | Importantly, |     | our | approach | is  | model- |
| -------- | --------------- | ------- | ---------- | ------- | --- | ------------ | ------------ | --- | --- | -------- | --- | ------ |
| Patel et | al., 2024; Kang | et al., | 2025) that | require |     |              |              |     |     |          |     |        |
fine-tuning or additional supervised datasets, we agnostic: it can be applied to various embedding
directlyoptimizetheembeddingoftheinputquery models, including multimodal encoders such as
atinferencetimewhilekeepingtheencoderfrozen. CLIP(Radfordetal.,2021),andenableseffective
multimodalretrievalwithoutadditionalfine-tuning
| LetE(·)denotetheencoder.        |     | Givenaninputquery |     |     |     |                          |     |     |     |     |     |     |
| ------------------------------- | --- | ----------------- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- |
| q,weobtainitsoriginalembedding: |     |                   |     |     |     | ortask-specificdatasets. |     |     |     |     |     |     |
|                                 |     | =E(q)∈Rd          |     |     |     | 4 Experiments            |     |     |     |     |     |     |
|                                 | e o |                   |     |     | (1) |                          |     |     |     |     |     |     |
|                                 |     |                   |     |     |     | 4.1 ExperimentalSetup    |     |     |     |     |     |     |
FromtheLLM-basedquerydecomposition,we
}K
obtain a set of positive sub-queries P = {p i Datasets and Metrics. We primarily focus on
i=1
| and a set | of negative | sub-queries | N = | {n }M | .   |          |                   |     |            |     |     |          |
| --------- | ----------- | ----------- | --- | ----- | --- | -------- | ----------------- | --- | ---------- | --- | --- | -------- |
|           |             |             |     | j     | j=1 | the task | of negation-aware |     | retrieval. |     | For | this, we |
Theirembeddingsarecomputedas:
|               |             |           |           |     |     | employ                                   | the NegConstraint |         |       | (Xu et    | al., 2025) | and   |
| ------------- | ----------- | --------- | --------- | --- | --- | ---------------------------------------- | ----------------- | ------- | ----- | --------- | ---------- | ----- |
|               |             |           |           |     |     | NevIR (Weller                            |                   | et al., | 2024) | datasets. | Following  |       |
|               | e =E(p      | ), e =E(n | ).        |     | (2) |                                          |                   |         |       |           |            |       |
|               | pi          | i nj      | j         |     |     | priorwork,weevaluateperformanceonNegCon- |                   |         |       |           |            |       |
|               |             |           |           |     |     | straint using                            | nDCG@10           |         | and   | MAP@100,  |            | while |
| We initialize | a learnable | query     | embedding |     | e u |                                          |                   |         |       |           |            |       |
|               |             |           |           |     |     | NevIR is                                 | evaluated         | using   | the   | Pairwise  | metric.    | In    |
withtheoriginalembedding.
addition,fornegation-awaretext-to-imageretrieval,
|     |     |            |     |     |     | we evaluate   | on  | a text-to-image |        | retrieval |          | dataset |
| --- | --- | ---------- | --- | --- | --- | ------------- | --- | --------------- | ------ | --------- | -------- | ------- |
|     |     | e u ←e o . |     |     | (3) |               |     |                 |        |           |          |         |
|     |     |            |     |     |     | that contains |     | negation,       | namely | the       | COCO-Neg |         |
Theobjectiveconsistsofthreecomponents: (i) from NegBench (Alhamoud et al., 2025), using
anattractiontermthatpullstheoptimizedembed- Recall@5astheevaluationmetric.
dingtowardthepositiveembeddings,(ii)arepul-
|     |     |     |     |     |     | Baselines. | Because |     | our approach |     | does | not re- |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | --- | ------------ | --- | ---- | ------- |
sion term that pushes it away from the negative quire fine-tuning or additional training data, it
embeddings,and(iii)aconsistencytermthatpre- can be directly applied to any embedding model.
| servesthesemanticsoftheoriginalquery. |     |     |     | Theloss |     |          |            |     |         |     |         |      |
| ------------------------------------- | --- | --- | --- | ------- | --- | -------- | ---------- | --- | ------- | --- | ------- | ---- |
|                                       |     |     |     |         |     | For text | retrieval, | we  | compare |     | against | BGE- |
functionisdefinedas:
M3,BGE-large-en-v1.5,andBGE-small-en-v1.5.
|     |     |              |     |     |     | For text-to-image |     | retrieval, |     | we consider |     | OpenAI |
| --- | --- | ------------ | --- | --- | --- | ----------------- | --- | ---------- | --- | ----------- | --- | ------ |
|     |     | 1 (cid:88) K |     |     |     |                   |     |            |     |             |     |        |
L(e )=λ · ∥e −e ∥2 CLIP (Radford et al., 2021), CLIP-laion400m,
|     | u p | u   | pi  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
K
|     |     | i=1        |         |     |     | CLIP-datacomp,andNegCLIP(Yuksekgonuletal., |     |          |     |      |           |     |
| --- | --- | ---------- | ------- | --- | --- | ------------------------------------------ | --- | -------- | --- | ---- | --------- | --- |
|     |     | M          |         |     | (4) | 2022)asbaselines.                          |     |          |     |      |           |     |
|     |     | 1 (cid:88) | ∥2      |     |     |                                            |     |          |     |      |           |     |
|     | −λ  | n · ∥e     | u −e nj |     |     |                                            |     |          |     |      |           |     |
|     |     | M          |         |     |     | Implementation                             |     | Details. | We  | used | the [CLS] | to- |
j=1
|     |     |              | ∥2. |     |     | kenrepresentationforallembeddingmodels,em- |     |     |     |     |     |     |
| --- | --- | ------------ | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
|     | +λ  | o ·∥e u −e o |     |     |     |                                            |     |     |     |     |     |     |
4

|     |     | NegConstraint |     |      | NevIR    |     |     |        | NegConstraint |         |     |
| --- | --- | ------------- | --- | ---- | -------- | --- | --- | ------ | ------------- | ------- | --- |
|     |     | MAP           |     | nDCG | Pairwise |     |     | Method | MAP           | nDCG@10 |     |
BGE-small-en-v1.5 0.6702 0.7372 0.1569 BGE-Sw/Qwen 0.7144 0.7705
| w/DEO |     | 0.7302 |     | 0.7795 | 0.1894 |     | BGE-Sw/GPT |     | 0.7302 | 0.7795 |     |
| ----- | --- | ------ | --- | ------ | ------ | --- | ---------- | --- | ------ | ------ | --- |
BGE-large-en-v1.5 0.6299 0.7139 0.2552 BGE-Lw/Qwen 0.6859 0.7553
| w/DEO  |     | 0.7327 |     | 0.7877 | 0.2776 |     | BGE-Lw/GPT   |     | 0.7327 | 0.7877 |     |
| ------ | --- | ------ | --- | ------ | ------ | --- | ------------ | --- | ------ | ------ | --- |
| BGE-M3 |     | 0.6374 |     | 0.7250 | 0.2668 |     | BGE-M3w/Qwen |     | 0.7280 | 0.7871 |     |
| w/DEO  |     | 0.7379 |     | 0.7946 | 0.2928 |     | BGE-M3w/GPT  |     | 0.7379 | 0.7946 |     |
Table1:PerformanceontheNegConstraint(Xuetal., Table 3: Performance comparison on the NegCon-
2025)benchmarkfornegationretrieval. “w/DEO” straintdatasetusingdifferentLLMbackbonesfor
representstheperformanceafterapplyingourproposed querydecomposition. QwenreferstoQwen2.5-1.5B-
method. AllnDCGscorescorrespondtonDCG@10. Instruct, and GPT refers to GPT-4.1-nano. BGE-S,
BGE-LdenoteBGE-small-en-v1.5,BGE-large-en-v1.5,
|     |            |     | Recall@5 |     |     | respectively. |     |     |     |     |     |
| --- | ---------- | --- | -------- | --- | --- | ------------- | --- | --- | --- | --- | --- |
|     | OpenAICLIP |     | 0.4792   |     |     |               |     |     |     |     |     |
0.5392
w/DEO
CLIP-laion400m 0.5248 The consistent improvements across both
|     | w/DEO         |     | 0.5737 |     |     | ranking-basedmetricsandpairwisenegationevalu- |     |     |     |     |     |
| --- | ------------- | --- | ------ | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- |
|     | CLIP-datacomp |     | 0.4984 |     |     |                                               |     |     |     |     |     |
ationsuggestthatDEOenhancesthemodel’sabil-
|     | w/DEO |     | 0.5513 |     |     |     |     |     |     |     |     |
| --- | ----- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
NegCLIP 0.6715 itytocorrectlyinterpretandretrieveundernegated
|     | w/DEO |     | 0.6980 |     |     |        |             |              |                    |         |        |
| --- | ----- | --- | ------ | --- | --- | ------ | ----------- | ------------ | ------------------ | ------- | ------ |
|     |       |     |        |     |     | query  | conditions. | Overall,     | these              | results | demon- |
|     |       |     |        |     |     | strate | that        | DEO improves | negation-sensitive |         | re-    |
Table2: PerformanceontheCOCO-Negbenchmark
trievalperformanceacrossdiverseembeddingmod-
fortext-to-imagenegationretrieval(Alhamoudetal.,
| 2025). “w/DEO”representstheperformanceafterap- |     |     |     |     |     | els. |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
plyingourproposedmethod.
4.2.2 NegationBenchmarkonText-to-Image
| ployed | the FAISS | library, | and | computed | cosine |     | Retrieval |     |     |     |     |
| ------ | --------- | -------- | --- | -------- | ------ | --- | --------- | --- | --- | --- | --- |
similarityforretrieval. Forquerydecompositionin Ourmethodisapplicableregardlessoftheembed-
ourmethod,weusedOpenAI’sGPT-4.1-nanowith dingmodelormodality. Toverifythis,weevaluate
| atemperatureof0.1. |     | Acrossalltext-onlydatasets, |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onCOCO-Neg(Alhamoudetal.,2025).
| we set the | hyperparameters |     | λ   | = 1, λ | = 1, and |     |                                         |     |     |     |     |
| ---------- | --------------- | --- | --- | ------ | -------- | --- | --------------------------------------- | --- | --- | --- | --- |
|            |                 |     | p   | n      |          |     | AsshowninTable2,DEOconsistentlyimproves |     |     |     |     |
λ = 0.2,anduse20optimizationsteps. Formulti- Recall@5acrossallfourCLIP-basedmodels. Ope-
o
modalretrievaldatasets,weinsteadsetλ p = 1.0, nAICLIPshowsthelargestgain,withRecall@5
| λ = 1.0,andλ |     | = 1.0. |     |     |     |                                           |     |                    |     |              |     |
| ------------ | --- | ------ | --- | --- | --- | ----------------------------------------- | --- | ------------------ | --- | ------------ | --- |
| n            | o   |        |     |     |     | increasingby6.00%overthebaseline,andCLIP- |     |                    |     |              |     |
|              |     |        |     |     |     | Datacomp                                  |     | and CLIP-laion400m |     | also achieve | im- |
4.2 MainResults
|     |     |     |     |     |     | provementsof5.29%and4.89%,respectively. |     |     |     |     | No- |
| --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | --- |
4.2.1 NegationBenchmark
tably,evenNegCLIP,whichisexplicitlyfine-tuned
Weevaluateretrievalperformanceonnegation-and for compositional understanding, improves from
exclusion-intensivetasksusingtheNegConstraint 0.6715to0.6980,confirmingthatDEOyieldsaddi-
andNevIRbenchmarks. AsshowninTable1,ap- tionalgainsevenontopofnegation-awaremodels.
plying DEO consistently improves performance Theseresultsconfirmthatourmethodgeneralizes
effectivelytomultimodalretrievalsettings.
| across all | BGE variants, |     | yielding | gains | in MAP, |     |     |     |     |     |     |
| ---------- | ------------- | --- | -------- | ----- | ------- | --- | --- | --- | --- | --- | --- |
nDCG@10,andpairwiseaccuracy.
4.3 AblationStudy
Acrossallmodels,DEOleadstoclearimprove-
mentsonNegConstraint,indicatingbetterretrieval To analyze the contribution of each component,
| undernegationconstraints. |     |     | Forexample,onBGE- |     |     |     |         |          |         |                      |     |
| ------------------------- | --- | --- | ----------------- | --- | --- | --- | ------- | -------- | ------- | -------------------- | --- |
|                           |     |     |                   |     |     | we  | conduct | ablation | studies | from 4 perspectives: |     |
large-en-v1.5, applying DEO increases MAP by thechoiceofLLM,balancingparameters(λ ,λ ,
o n
+0.1028 (+16.32%) and nDCG@10 by +0.0738 λ ), the effect of query decomposition, and the
p
(+10.34%). Thesegainsarefurthersupportedbyre- optimizationprocedure.
sultsonNevIR,whichdirectlyevaluatesnegation-
4.3.1 EffectofDifferentLLMs
| awarepairwisediscrimination. |     |     |     | Inparticular,DEO |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
improves the NevIR pairwise score from 0.1569 To evaluate how an LLM’s decomposition abil-
to0.1894forBGE-small-en-v1.5,from0.2552to ity impacts our method, we conduct experiments
0.2776forBGE-large-en-v1.5,andfrom0.2668to with different backbone models. Specifically,
| 0.2928forBGE-M3. |     |     |     |     |     | wecompareGPT-4.1-nanoagainstQwen2.5-1.5B- |     |     |     |     |     |
| ---------------- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- |
5

COCO-Neg
|          |                      | Method     |     | Recall@5 |          |     |     |     |     |     |     |
| -------- | -------------------- | ---------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- |
|          | CLIP-laion400mw/Qwen |            |     | 0.5656   |          |     |     |     |     |     |     |
|          | CLIP-laion400mw/GPT  |            |     | 0.5737   |          |     |     |     |     |     |     |
|          | NegCLIPw/Qwen        |            |     | 0.6872   |          |     |     |     |     |     |     |
|          | NegCLIPw/GPT         |            |     | 0.6980   |          |     |     |     |     |     |     |
| Table 4: | Performance          | comparison |     | on       | COCO-Neg |     |     |     |     |     |     |
usingdifferentLLMbackbonesforquerydecompo-
sition.
|             |     | NegConstraint |         | COCO-Neg |        |     |     |     |     |     |     |
| ----------- | --- | ------------- | ------- | -------- | ------ | --- | --- | --- | --- | --- | --- |
| Parameters  |     | MAP           | nDCG@10 | Recall@5 |        |     |     |     |     |     |     |
| Baseline    |     | 0.6374        | 0.7250  |          | 0.4792 |     |     |     |     |     |     |
| 0.2,1.0,1.0 |     | 0.7379        | 0.7946  |          | 0.5349 |     |     |     |     |     |     |
| 0.2,1.0,2.0 |     | 0.7366        | 0.7942  |          | 0.5275 |     |     |     |     |     |     |
| 0.2,2.0,1.0 |     | 0.7160        | 0.7769  |          | 0.5237 |     |     |     |     |     |     |
| 1.0,1.0,1.0 |     | 0.6713        | 0.7510  |          | 0.5392 |     |     |     |     |     |     |
Figure2: Retrievalperformancewithrespecttothe
| 1.0,1.0,2.0 |     | 0.7034 | 0.7740 |     | 0.5303 |     |     |     |     |     |     |
| ----------- | --- | ------ | ------ | --- | ------ | --- | --- | --- | --- | --- | --- |
numberofoptimizationstepsonNegConstraint.
| 1.0,2.0,1.0 |     | 0.7013 | 0.7712 |     | 0.5246 |     |     |     |     |     |     |
| ----------- | --- | ------ | ------ | --- | ------ | --- | --- | --- | --- | --- | --- |
Table5: Effectofvaryingtheweightbalancingpa-
| rametersλ | ,λ                                | ,andλ | ontheNegConstraintand |     |     |     |     |     |     |     |     |
| --------- | --------------------------------- | ----- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|           | o                                 | p n   |                       |     |     |     |     |     |     |     |     |
| COCO-Neg. | NegConstraintresultsuseBGE-M3,and |       |                       |     |     |     |     |     |     |     |     |
COCO-NegresultsuseOpenAICLIP.Optimizationstep
is20.
NegConstraint
|                         | Method                               |         | MAP              |     | nDCG@10 |     |     |     |     |     |     |
| ----------------------- | ------------------------------------ | ------- | ---------------- | --- | ------- | --- | --- | --- | --- | --- | --- |
| BGE-M3                  |                                      |         | 0.6374           |     | 0.7250  |     |     |     |     |     |     |
| Ours(onlydecompose,AVG) |                                      |         | 0.6451           |     | 0.7312  |     |     |     |     |     |     |
| Ours(onlydecompose,RRF) |                                      |         | 0.6641           |     | 0.7417  |     |     |     |     |     |     |
| Ours(full)              |                                      |         | 0.7379           |     | 0.7946  |     |     |     |     |     |     |
| Table 6:                | Ablation                             | results | on NegConstraint |     | with    |     |     |     |     |     |     |
| BGE-M3.                 | Querydecompositionaloneyieldslimited |         |                  |     |         |     |     |     |     |     |     |
gains,whilemostperformanceimprovementsarisefrom
Figure3: Retrievalperformancewithrespecttothe
embeddingoptimizationratherthandecompositionit-
| self.                                        |       |                |           |       |           | numberofoptimizationstepsonCOCO-Neg. |          |      |               |              |     |
| -------------------------------------------- | ----- | -------------- | --------- | ----- | --------- | ------------------------------------ | -------- | ---- | ------------- | ------------ | --- |
|                                              |       |                |           |       |           | of0.7946bysettingλ                   |          |      | = 0.2.        |              |     |
| Instruct                                     | (Team | et al., 2024), | which     | has   | far fewer |                                      |          | o    |               |              |     |
|                                              |       |                |           |       |           | In contrast,                         | COCO-Neg |      | peaked        | at 0.5392    | Re- |
| parameters.                                  | As    | shown          | in Tables | 3 and | 4, GPT-   |                                      |          |      |               |              |     |
|                                              |       |                |           |       |           | call@5 with                          | λ =      | 1.0, | as preserving | the original |     |
| 4.1-nanoconsistentlyoutperformsQwen2.5-1.5B- |       |                |           |       |           |                                      | o        |      |               |              |     |
InstructacrossallembeddingmodelsonbothNeg- semanticcontextisessentialtomaintainalignment
insharedvision-languagespaceslikeCLIP.These
ConstraintandCOCO-Neg,whichweattributeto
findingsindicatethatDEOgeneralizeseffectively
moreprecisequerydecompositionsfromthelarger
acrossdiversemodalitiesthroughitsadaptiveopti-
| model.    | Nevertheless, |          | even with | Qwen2.5-1.5B- |          |                    |     |     |     |     |     |
| --------- | ------------- | -------- | --------- | ------------- | -------- | ------------------ | --- | --- | --- | --- | --- |
| Instruct, | our method    | achieves |           | notable       | improve- | mizationobjective. |     |     |     |     |     |
mentsoverthebaselines,indicatingthatDEOde-
4.3.3 EffectofQueryDecomposition
liversconsistentgainsregardlessofLLMscale.
Toanalyzetheroleofquerydecomposition,wecon-
4.3.2 EffectofWeightBalancing ductanablationstudyonthedecompositionstep.
Weanalyzetheimpactoftheweightbalancingpa- Table6reportsresultsontheNegConstraintdataset
rameters λ (original consistency), λ (positive), withBGE-M3. WeintroduceanOnlyDecompose
|     | o   |     |     |     | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
and λ n (negative) across both the NegConstraint variantthatretrievesusingtheaveragedembedding
andCOCO-Negbenchmarks. AsshowninTable5, ofdecomposedpositiveandnegativesub-queries,
DEOconsistentlyoutperformsthebaselineacross and an Only Decompose (RRF) variant that re-
alltestedconfigurations,demonstratingitsinherent trieves separately for each sub-query and merges
robustness to hyperparameter variations. On the resultsusingReciprocalRankFusion(RRF)with
text-onlyNegConstraintdataset,thebestconfigura- k=60. AsshowninTable6,querydecomposition
tionreachedaMAPof0.7379andannDCG@10 alone yields only limited improvements over the
6

Figure5: Trajectoryofoptimizedqueryembedding
|     |     |     |     | e in PCA-projected | space. | The initial | embedding |
| --- | --- | --- | --- | ------------------ | ------ | ----------- | --------- |
u
|     |     |     |     | (black•)movestowardthefinalstate(blue•). |               |                  | Positive   |
| --- | --- | --- | --- | ---------------------------------------- | ------------- | ---------------- | ---------- |
|     |     |     |     | examples                                 | (green △) and | the ground-truth | (yellow ⋆) |
actasattractors,whilenegativeexamples(red×)exert
|     |     |     |     | repellingforces. | Othercorpusembeddingsareshown |     |     |
| --- | --- | --- | --- | ---------------- | ----------------------------- | --- | --- |
inlightgray.
5 Analysis
5.1 QualityofQueryDecomposition
WeanalyzehowtheLLMdecomposesqueriesinto
|     |     |     |     | positiveandnegativesub-queries. |     | Toevaluatede- |     |
| --- | --- | --- | --- | ------------------------------- | --- | ------------- | --- |
compositionquality,weperformabinarycorrect-
Figure4: Examplesofquerydecomposition. InNeg- nessassessmentonNegConstraint(Xuetal.,2025)
| Constraint | (a) and COCO-Neg | (b), the LLM | decom- |     |     |     |     |
| ---------- | ---------------- | ------------ | ------ | --- | --- | --- | --- |
usingGPT-4.1-mini,measuringwhethereachout-
posesqueriesintopositivesub-queriescapturingdesired
|     |     |     |     | put captures | the intended | positive | and negative |
| --- | --- | --- | --- | ------------ | ------------ | -------- | ------------ |
elementsandnegativesub-queriestargetingexcluded
|     |     |     |     | components | of the original | query. The | decompo- |
| --- | --- | --- | --- | ---------- | --------------- | ---------- | -------- |
concepts.
|     |     |     |     | sition achieves | 91.76% | accuracy, indicating | that |
| --- | --- | --- | --- | --------------- | ------ | -------------------- | ---- |
generatedsub-queriesarelargelyalignedwithuser
| baseline. | WhileRRF-basedaggregationprovides |     |     |     |     |     |     |
| --------- | --------------------------------- | --- | --- | --- | --- | --- | --- |
additionalgains, theperformanceremainsfarbe- intent. Inaddition,weempiricallyverifythatthe
low that of the full method. This indicates that inputqueriesareeffectivelydecomposedonboth
theperformanceimprovementsdonotstemfrom NegConstraintandCOCO-Neg(Alhamoudetal.,
2025),asillustratedinFigure4.
decompositionitself,butfromthesubsequentem-
beddingoptimization.
5.2 EmbeddingSpaceAnalysis
4.3.4 EffectofOptimizationSteps Wevisualizehowtheoptimizedqueryembedding
Weanalyzetheeffectofthenumberofoptimization e movesrelativetopositives(e ),negatives(e ),
|     |     |     |     | u   |     | p   | n   |
| --- | --- | --- | --- | --- | --- | --- | --- |
stepsonretrievalperformance. OnNegConstraint and corpus embeddings. We fit a PCA on the
(Figure 2), performance improves sharply when full corpus embeddings of BGE-M3 and project
increasingstepsfrom0to20,andremainsstable thequerytrajectory,decomposedsub-queries,and
between 20 and 50 steps. However, beyond 100 ground-truth documents onto the same 2D space.
steps,bothnDCG@10andMAP@100gradually ExperimentsareconductedonNegConstraint(Xu
decline. OnCOCO-Neg(Figure3),Recall@5sim- etal.,2025)withthesamesettingasSec4.1.
ilarly improves from 0 to 20 steps, peaks around Figure 5 illustrates a representative example
50steps,andslightlydecreasesbeyond100steps. where the query asks for journalistic evaluation
Inbothcases,20to50stepsissufficienttoachieve criteria while excluding their sources and the
strongperformance,andweadopt20stepsasthe Pulitzer Prize for History. The initial embed-
defaultsettingacrossallexperiments. ding (black) progressively shifts toward the final
7

|     |     |     |     | both CPU | and | GPU environments. |     |     | On a CPU |
| --- | --- | --- | --- | -------- | --- | ----------------- | --- | --- | -------- |
(AMDRyzen75800X8-CoreProcessor,64.0GB
RAM),DEOwith20optimizationstepsrequireda
totalof0.016seconds(average0.000665seconds
|     |     |     |     | per step), | while    | 50 steps | required | 0.035  | seconds  |
| --- | --- | --- | --- | ---------- | -------- | -------- | -------- | ------ | -------- |
|     |     |     |     | (average   | 0.000640 | seconds  | per      | step). | On a GPU |
(NVIDIAGeForceRTX306012GB),DEOwith
20optimizationstepsrequiredatotalof0.033sec-
onds(average0.00172secondsperstep),while50
|     |     |     |     | steps required   |         | 0.095 seconds               |           | (average | 0.001932 |
| --- | --- | --- | --- | ---------------- | ------- | --------------------------- | --------- | -------- | -------- |
|     |     |     |     | secondsperstep). |         | Theseresultsdemonstratethat |           |          |          |
|     |     |     |     | our method       | remains | highly                      | efficient | across   | both     |
CPUandGPUsettings,makingitpracticalforreal-
Figure 6: Optimization trajectory of e in PCA- worldapplicationswhereGPUresourcesmaybe
u
projected CLIP space. Initial (black •) embedding limitedoradditionaltrainingdataisunavailable.
evolvestowardfinalstate(blue•),attractedbypositives
(green△)andground-truth(yellow⋆)whilerepelled
6 Conclusion
| bynegatives(red×). |     | Graypointsdenotethecorpus. |     |     |     |     |     |     |     |
| ------------------ | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- |
optimized point (blue), attracted by positive sub- Inthiswork,weproposeDirectEmbeddingOpti-
queries (green triangles) and the gold document mization(DEO),asimpleyeteffectivemethodfor
(yellow star), while repelled from negative sub- negation-andexclusion-awareretrievalthatdoes
| queries(redcrosses). |     | Inthebaseline,adocument |     |             |             |     |            |           |     |
| -------------------- | --- | ----------------------- | --- | ----------- | ----------- | --- | ---------- | --------- | --- |
|                      |     |                         |     | not require | fine-tuning | or  | additional | datasets. | By  |
aboutthePulitzerPrizeforHistory—whichshould decomposinguserqueriesintopositiveandnega-
beexcluded—ranks1st,pushingtheground-truth tivesub-queriesanddirectlyoptimizingtheembed-
| torank6. | Afteroptimization,theground-truthrises |     |     |     |     |     |     |     |     |
| -------- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
dingspacewithcontrastiveloss,DEOalignsquery
to rank 1 (NDCG@10: 0.356→1.0). Across the embeddingsmorepreciselywithuserintent. Our
top-5 improved queries, ground-truth documents experiments demonstrate that DEO consistently
movefromranks6–33intothetop2,withanaver- outperformsbaselinemethodsacrossbothtextand
| ageNDCG@10gainof+0.63. |     | Thisconfirmsthat |     |            |           |        |           |     |             |
| ---------------------- | --- | ---------------- | --- | ---------- | --------- | ------ | --------- | --- | ----------- |
|                        |     |                  |     | multimodal | retrieval | tasks, | achieving |     | substantial |
the contrastive optimization effectively reshapes gains on benchmarks involving negation and ex-
E u towardrelevantregionswhilesuppressingneg- clusion. Theseresultshighlighttherobustnessand
atives.
|     |     |     |     | practicality | of our | approach | in  | real-world | scenar- |
| --- | --- | --- | --- | ------------ | ------ | -------- | --- | ---------- | ------- |
We further extend this analysis to the vision- ios where negation and exclusion are frequently
language setting. Using CLIP ViT-B/32 on the presentinuserqueries.
| COCO negated   | retrieval | benchmark       | (Alhamoud |     |     |     |     |     |     |
| -------------- | --------- | --------------- | --------- | --- | --- | --- | --- | --- | --- |
| et al., 2025), | we fit    | PCA on the CLIP | image em- |     |     |     |     |     |     |
7 Limitation
beddingsandprojectthetextquerytrajectoryonto
thesame2Dspace. Figure6showsthatthesame While DEO proves effective without fine-tuning,
| trajectorypatternholds: |     | theinitialtextembedding |     |           |        |         |         |              |     |
| ----------------------- | --- | ----------------------- | --- | --------- | ------ | ------- | ------- | ------------ | --- |
|                         |     |                         |     | it relies | on the | ability | of LLMs | to correctly | de- |
(black)shiftstowardtheground-truthimage(yel- compose user queries into positive and negative
lowstar)andpositivesub-queries(greentriangles), sub-queries. As shown in the Sec 4.3, the final
whilemovingawayfromnegativesub-queries(red retrievalperformancemayvarydependingonthe
crosses) that encode the negated concept. This decompositionqualityoftheLLM.Webelievethat
demonstratesthatDEOgeneralizesacrossbothtext DEOprovidesapromisingdirectionforbuilding
retrieval and cross-modal image retrieval, effec- lightweightandcontrollableretrievalsystems. Fu-
| tively reshaping | the | query embedding | in CLIP’s |     |     |     |     |     |     |
| ---------------- | --- | --------------- | --------- | --- | --- | --- | --- | --- | --- |
tureworkcouldexploreenhancingquerydecom-
sharedvision-languagespacetosuppressnegated position with more robust LLMs, incorporating
| attributes. |     |     |     | adaptiveoptimizationstrategiesthatautomatically |           |            |     |     |            |
| ----------- | --- | --- | --- | ----------------------------------------------- | --------- | ---------- | --- | --- | ---------- |
|             |     |     |     | select loss                                     | balancing | parameters |     | per | query, and |
5.3 ComputationalEfficiency extendingDEOtodiversemultimodaldatasetsbe-
Toevaluatethecomputationaloverheadofourpa- yondimages,suchasaudio.
| rameter optimization, |     | we measured | runtime on |     |     |     |     |     |     |
| --------------------- | --- | ----------- | ---------- | --- | --- | --- | --- | --- | --- |
8

References
JunnanLi,DongxuLi,SilvioSavarese,andStevenHoi.
|                  |        |             |     |          |     | 2023. Blip-2: | Bootstrapping     | language-image |     | pre-       |
| ---------------- | ------ | ----------- | --- | -------- | --- | ------------- | ----------------- | -------------- | --- | ---------- |
| Kumail Alhamoud, | Shaden | Alshammari, |     | Yonglong |     |               |                   |                |     |            |
|                  |        |             |     |          |     | training      | with frozen image | encoders       | and | large lan- |
Tian, Guohao Li, Philip HS Torr, Yoon Kim, and guage models. In International conference on ma-
| MarzyehGhassemi.2025. |     | Vision-languagemodels |     |     |     |     |     |     |     |     |
| --------------------- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
chinelearning,pages19730–19742.PMLR.
| donotunderstandnegation. |            | InProceedingsofthe |             |         |     |            |                    |        |     |            |
| ------------------------ | ---------- | ------------------ | ----------- | ------- | --- | ---------- | ------------------ | ------ | --- | ---------- |
| Computer                 | Vision and | Pattern            | Recognition | Confer- |     |            |                    |        |     |            |
|                          |            |                    |             |         |     | Junnan Li, | Dongxu Li, Caiming | Xiong, |     | and Steven |
ence,pages29612–29622.
|     |     |     |     |     |     | Hoi.2022. | Blip: Bootstrappinglanguage-imagepre- |     |               |     |
| --- | --- | --- | --- | --- | --- | --------- | ------------------------------------- | --- | ------------- | --- |
|     |     |     |     |     |     | training  | for unified vision-language           |     | understanding |     |
StanislawAntol,AishwaryaAgrawal,JiasenLu,Mar-
|     |     |     |     |     |     | andgeneration. | InInternationalconferenceonma- |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------- | ------------------------------ | --- | --- | --- |
garetMitchell,DhruvBatra,CLawrenceZitnick,and chinelearning,pages12888–12900.PMLR.
| DeviParikh.2015. | Vqa: | Visualquestionanswering. |     |     |     |     |     |     |     |     |
| ---------------- | ---- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
InProceedingsoftheIEEEinternationalconference Maitreya Patel, Naga Sai Abhiram Kusumba, Sheng
oncomputervision,pages2425–2433. Cheng,ChanghoonKim,TejasGokhale,ChittaBaral,
|     |     |     |     |     |     | and1others.2024. | Tripletclip: | Improvingcomposi- |     |     |
| --- | --- | --- | --- | --- | --- | ---------------- | ------------ | ----------------- | --- | --- |
PayalBajaj,DanielCampos,NickCraswell,LiDeng,
tionalreasoningofclipviasyntheticvision-language
JianfengGao,XiaodongLiu,RanganMajumder,An-
|                |         |        |     |         |     | negatives. | Advancesinneuralinformationprocess- |     |     |     |
| -------------- | ------- | ------ | --- | ------- | --- | ---------- | ----------------------------------- | --- | --- | --- |
| drew McNamara, | Bhaskar | Mitra, | Tri | Nguyen, | and |            |                                     |     |     |     |
ingsystems,37:32731–32760.
| 1others.2016. | Msmarco: | Ahumangeneratedma- |     |     |     |     |     |     |     |     |
| ------------- | -------- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
chinereadingcomprehensiondataset. arXivpreprint AlecRadford,JongWookKim,ChrisHallacy,Aditya
arXiv:1611.09268. Ramesh,GabrielGoh,SandhiniAgarwal,GirishSas-
try,AmandaAskell,PamelaMishkin,JackClark,and
YuliangCai,JesseThomason,andMohammadRostami.
|                                  |                                 |     |     |               |     | 1others.2021.                   | Learningtransferablevisualmodels |     |                 |     |
| -------------------------------- | ------------------------------- | --- | --- | ------------- | --- | ------------------------------- | -------------------------------- | --- | --------------- | --- |
| 2025. Tng-clip:                  | Training-timenegationdatagener- |     |     |               |     |                                 |                                  |     |                 |     |
|                                  |                                 |     |     |               |     | fromnaturallanguagesupervision. |                                  |     | InInternational |     |
| ationfornegationawarenessofclip. |                                 |     |     | arXivpreprint |     |                                 |                                  |     |                 |     |
conferenceonmachinelearning,pages8748–8763.
| arXiv:2505.18434. |     |     |     |     |     | PmLR. |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
PeiranDong,SongGuo,JunxiaoWang,BingjieWang, KirillSergeevichShevkunov,AndreyPloskonosov,and
JieweiZhang,andZimingLiu.2023. Towardstest- LiudmilaProkhorenkova. Relevance-basedembed-
| time refusals | via concept | negation. |     | Advances | in  |     |     |     |     |     |
| ------------- | ----------- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- |
dingsforefficientrelevanceretrieval.
NeuralInformationProcessingSystems,36:26638–
26649.
RiturajSingh,RahulKumar,andVivekSridhar.2023.
|     |     |     |     |     |     | Nlms: Augmentingnegationinlanguagemodels. |     |     |     | In  |
| --- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- |
AaronGrattafiori,AbhimanyuDubey,AbhinavJauhri,
FindingsoftheAssociationforComputationalLin-
Abhinav Pandey, Abhishek Kadian, Ahmad Al- guistics: EMNLP2023,pages13104–13116.
Dahle,AieshaLetman,AkhilMathur,AlanSchelten,
AlexVaughan,and1others.2024. Thellama3herd QwenTeamand1others.2024. Qwen2technicalreport.
ofmodels. arXivpreprintarXiv:2407.21783. arXivpreprintarXiv:2407.10671,2:3.
HaoKang,TevinWang,andChenyanXiong.2025. In- LiangWang,NanYang,XiaolongHuang,LinjunYang,
terpretandcontroldenseretrievalwithsparselatent Rangan Majumder, and Furu Wei. 2024. Improv-
InProceedingsofthe2025Conferenceof
| features. |     |     |     |     |     | ingtextembeddingswithlargelanguagemodels. |     |     |     | In  |
| --------- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- |
theNationsoftheAmericasChapteroftheAssoci- Proceedingsofthe62ndAnnualMeetingoftheAs-
ation for Computational Linguistics: Human Lan- sociationforComputationalLinguistics(Volume1:
guageTechnologies(Volume2: ShortPapers),pages LongPapers),pages11897–11916.
700–709.
OrionWeller,DawnLawrie,andBenjaminVanDurme.
VladimirKarpukhin,BarlasOguz,SewonMin,Patrick 2024. Nevir:Negationinneuralinformationretrieval.
Lewis,LedellWu,SergeyEdunov,DanqiChen,and InProceedingsofthe18thConferenceoftheEuro-
Wen-tauYih.2020. Densepassageretrievalforopen- peanChapteroftheAssociationforComputational
domainquestionanswering. InProceedingsofthe Linguistics (Volume 1: Long Papers), pages 2274–
| 2020 conference | on empirical |     | methods | in natural |     | 2287. |     |     |     |     |
| --------------- | ------------ | --- | ------- | ---------- | --- | ----- | --- | --- | --- | --- |
languageprocessing(EMNLP),pages6769–6781.
GanlinXu,ZhoujiaZhang,WangyiMei,JiaqingLiang,
Jung-JunKim,Dong-GyuLee,JialinWu,Hong-Gyu WeijiaLu,XiaodongZhang,ZhifeiYang,Xiaofeng
| Jung,andSeong-WhanLee.2021. |     |     | Visualquestion |     |     |                                    |     |     |     |         |
| --------------------------- | --- | --- | -------------- | --- | --- | ---------------------------------- | --- | --- | --- | ------- |
|                             |     |     |                |     |     | Ma,YanghuaXiao,andDeqingYang.2025. |     |     |     | Logical |
answeringbasedonlocal-scene-awarereferringex- consistencyisvital: Neural-symbolicinformationre-
pressiongeneration. NeuralNetworks,139:158–167. trievalfornegative-constraintqueries. InFindingsof
|     |     |     |     |     |     | theAssociationforComputationalLinguistics: |     |     |     | ACL |
| --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- |
PatrickLewis,EthanPerez,AleksandraPiktus,Fabio 2025,pages1828–1847.
Petroni,VladimirKarpukhin,NamanGoyal,Hein-
richKüttler, MikeLewis, Wen-tauYih, TimRock- AnYang,AnfengLi,BaosongYang,BeichenZhang,
täschel,and1others.2020. Retrieval-augmentedgen- Binyuan Hui, Bo Zheng, Bowen Yu, Chang
erationforknowledge-intensivenlptasks. Advances Gao, Chengen Huang, Chenxu Lv, and 1 others.
inneuralinformationprocessingsystems,33:9459– 2025. Qwen3 technical report. arXiv preprint
| 9474. |     |     |     |     |     | arXiv:2505.09388. |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- |
9

MertYuksekgonul,FedericoBianchi,PratyushaKalluri,
| Dan Jurafsky,       | and      | James       | Zou. 2022. | When  | and      |
| ------------------- | -------- | ----------- | ---------- | ----- | -------- |
| why vision-language |          | models      | behave     | like  | bags-of- |
| words,              | and what | to do about | it?        | arXiv | preprint |
arXiv:2210.01936.
| Sepanta       | Zeighami,   | Zac           | Wellmer,    | and | Aditya     |
| ------------- | ----------- | ------------- | ----------- | --- | ---------- |
| Parameswaran. |             | 2024. Nudge:  | Lightweight |     | non-       |
| parametric    | fine-tuning | of embeddings |             | for | retrieval. |
arXivpreprintarXiv:2409.02343.
10
