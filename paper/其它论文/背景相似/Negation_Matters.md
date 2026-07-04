This CVPR Workshop paper is the Open Access version, provided by the Computer Vision
Foundation. Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
Negation Matters: Training-Free Negation-Aware Image Retrieval
AashishPokhrelandShivanandVenkannaSheshappanavar
GeometricIntelligenceResearchLab.,Dept. ofElectricalEngineeringandComputerScience
UniversityofWyoming,USA
|     |     |     | {apokhrel, ssheshap}@uwyo.edu |     |     |     |     |     |     |
| --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- |
Figure1. Top-5retrievalresultsforthenegatedquery“Animageofadognotonabeach.” Redbordersindicateincorrectretrievals
(imagescontainingabeachorsand);greenbordersindicatecorrectretrievals(non-beachscenes). TheCLIPbaseline[24]ranksbeach
imagesatallfivepositions,failingtorespectthenegationconstraint. OurSpaceVLM-DRCcorrectlyexcludesthenegatedconceptacross
alltop-5rankswithoutanyadditionaltraining.
Abstract frozen CLIP backbone. This framework first decomposes
|     |     |     |     | queries  | into affirmative, | negated, | and counterfactual |         | com-    |
| --- | --- | --- | --- | -------- | ----------------- | -------- | ------------------ | ------- | ------- |
|     |     |     |     | ponents, | then applies      | dynamic  | repulsion          | to push | negated |
Negation—the linguistic ability to assert the absence of a conceptsawayintheembeddingspace,andfinallyanchors
concept—isacriticalbottleneckinvision-languageunder-
retrievalwithinthefull-captioncontexttopreserveseman-
standing. Vision-language models have demonstrated re- tic coherence. SpaceVLM-DRC surpasses state-of-the-art
markable success in aligning visual and textual represen- results on the MSRVTT negation retrieval benchmark and
| tation across | domains such | as content moderation, | medi- |          |             |            |          |            |     |
| ------------- | ------------ | ---------------------- | ----- | -------- | ----------- | ---------- | -------- | ---------- | --- |
|               |              |                        |       | achieves | performance | comparable | to fully | fine-tuned | ap- |
cal image retrieval, and natural language-guided search. proachesontheCOCOnegatedretrievaldataset.Crucially,
Yet, these models consistently fail to handle negation ro- it requires no model retraining while preserving zero-shot
| bustly, often | retrieving | images that contain | the very con- |                |     |             |          |          |           |
| ------------- | ---------- | ------------------- | ------------- | -------------- | --- | ----------- | -------- | -------- | --------- |
|               |            |                     |               | generalization | on  | non-negated | queries. | Our code | is avail- |
ceptthatwasexplicitlyexcluded. Existingmethodsaddress able at https://github.com/aashishpokhrel27/spacevlm-drc.
| this limitation | through  | fine-tuning on          | synthetic negation |     |     |     |     |     |     |
| --------------- | -------- | ----------------------- | ------------------ | --- | --- | --- | --- | --- | --- |
| corpora—an      | approach | that is computationally | expensive,         |     |     |     |     |     |     |
dataset-dependent,andpronetocompromisinggeneraliza-
| tiontounseendistributions. |     | WeproposeSpaceVLM-DRC, |     |     |     |     |     |     |     |
| -------------------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
1.Introduction
whichintroducesDynamicRepulsionwithContextAnchor-
ing (DRC) into the SpaceVLM framework as a training- Vision-language models (VLMs) have become central
free inference time negation resolution mechanism over a to modern multimodal applications, including image re-
11353

trieval [6, 10, 24], visual question answering [2, 6], and butlimitedinsightsintomodelunreliability[3,28]. These
natural-language-guided decision-making in domains such benchmarks reveal the unreliability of current VLMs un-
as healthcare [8], robotics [18], and content manage- der negation, but their limited scale and diversity hinder
ment[6].Byjointlyencodingvisualandtextualinput,these the development of systematic methods and fair compar-
| VLMs enable       | users | to         | query | large visual  | collections |           | using | isons[3,22]. |        |              |               |     |       |
| ----------------- | ----- | ---------- | ----- | ------------- | ----------- | --------- | ----- | ------------ | ------ | ------------ | ------------- | --- | ----- |
| natural language, |       | supporting |       | context-aware |             | responses | that  |              |        |              |               |     |       |
|                   |       |            |       |               |             |           |       | In this      | paper, | we introduce | SpaceVLM-DRC, |     | which |
gofarbeyondtraditionalvisionpipelines[4,16,17,21].As
|     |     |     |     |     |     |     |     | incorporates | Dynamic | Repulsion | with | Context | Anchoring |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | --------- | ---- | ------- | --------- |
VLMsaredeployedinincreasinglysafety-criticalanduser-
intotheSpaceVLMframeworkasatraining-free,inference-
| facing systems |            | [1, 15,      | 20], their | ability | to         | correctly | follow |               |                                          |                      |     |      |               |
| -------------- | ---------- | ------------ | ---------- | ------- | ---------- | --------- | ------ | ------------- | ---------------------------------------- | -------------------- | --- | ---- | ------------- |
|                |            |              |            |         |            |           |        | time negation |                                          | resolution mechanism |     | over | a frozen CLIP |
| nuanced        | linguistic | instructions |            | becomes | essential. |           | Within |               |                                          |                      |     |      |               |
|                |            |              |            |         |            |           |        | backbone.     | Alightweightlanguagemodelfirstdecomposes |                      |     |      |               |
thislandscape,negationstandsoutasaparticularlyimpor-
eachqueryintoitsaffirmative,negated,andcounterfactual
tantyetunderexploredcapability[3,5].
|      |           |         |        |     |      |         |        | components. |     | Dynamic repulsion |     | then adaptively | pushes |
| ---- | --------- | ------- | ------ | --- | ---- | ------- | ------ | ----------- | --- | ----------------- | --- | --------------- | ------ |
| Many | realistic | queries | depend | not | only | on what | should |             |     |                   |     |                 |        |
negatedconceptsawayintheimageembeddingspace,with
| be present   | in a | scene        | but also | on    | what must | be      | explic- |           |        |             |            |         |           |
| ------------ | ---- | ------------ | -------- | ----- | --------- | ------- | ------- | --------- | ------ | ----------- | ---------- | ------- | --------- |
|              |      |              |          |       |           |         |         | exclusion | scaled | to negation | intensity. | Context | anchoring |
| itly absent, | such | as “retrieve |          | scans | showing   | a tumor | but     |           |        |             |            |         |           |
finallyenrichestheaffirmativesignalwithfull-captioncon-
| without calcification,” |     |          | “show  | street | scenes  | with    | cars but |                  |     |                    |     |            |            |
| ----------------------- | --- | -------- | ------ | ------ | ------- | ------- | -------- | ---------------- | --- | ------------------ | --- | ---------- | ---------- |
|                         |     |          |        |        |         |         |          | text, preserving |     | semantic coherence |     | throughout | retrieval. |
| no pedestrians,”        |     | or “find | images | of     | a beach | without | peo-     |                  |     |                    |     |            |            |
Ourmaincontributionsareasfollows:
| ple.” These | are | natural, | well-formed |     | requests—yet |     | current |     |     |     |     |     |     |
| ----------- | --- | -------- | ----------- | --- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- |
VLMs routinely fail them. Such failures produce quali- • Training-free negation resolution: We propose
|                    |     |            |      |      |     |        |        | SpaceVLM-DRC, |     | a frozen-backbone |     | framework | that |
| ------------------ | --- | ---------- | ---- | ---- | --- | ------ | ------ | ------------- | --- | ----------------- | --- | --------- | ---- |
| tatively incorrect |     | retrievals | even | when | the | models | appear |               |     |                   |     |           |      |
strongonstandardbenchmarks, directlyunderminingtrust resolves negation purely at inference time, requiring
insafety-criticalapplicationssuchasmedicalimaging[8], no gradient updates, synthetic corpora, or weight
| autonomoussystems[18],andcontentfiltering[6]. |         |               |     |     |             |             |      | modifications.   |     |                       |     |             |            |
| --------------------------------------------- | ------- | ------------- | --- | --- | ----------- | ----------- | ---- | ---------------- | --- | --------------------- | --- | ----------- | ---------- |
|                                               |         |               |     |     |             |             |      | • Inference-time |     | decomposition         |     | and dynamic | control:   |
| TherootcauseliesinthetrainingofVLMs[22].      |         |               |     |     |             |             | Mod- |                  |     |                       |     |             |            |
|                                               |         |               |     |     |             |             |      | We introduce     |     | a query decomposition |     | pipeline    | that sepa- |
| els such                                      | as CLIP | are optimized |     | on  | large-scale | collections |      |                  |     |                       |     |             |            |
ofpositiveimage-textpairs,implicitlyreinforcingassocia- ratesaffirmative,negated,andcounterfactualcomponents
|     |     |     |     |     |     |     |     | and dynamically |     | scales repulsion |     | and exclusion | regions |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ---------------- | --- | ------------- | ------- |
tionsbetweenvisualcontentanddescriptivelanguage[24].
basedonnegationstrength.
| Negation | words | such as | “no,” | “not,” | and “without” |     | appear |     |     |     |     |     |     |
| -------- | ----- | ------- | ----- | ------ | ------------- | --- | ------ | --- | --- | --- | --- | --- | --- |
infrequentlyinthepretrainingcorpora(≤0.7%ofthecap- • Strong empirical performance with zero-shot preser-
|            |           |            |     |         |     |          |      | vation: | SpaceVLM-DRC |     | surpasses |     | state-of-the-art |
| ---------- | --------- | ---------- | --- | ------- | --- | -------- | ---- | ------- | ------------ | --- | --------- | --- | ---------------- |
| tions) and | are never | explicitly |     | modeled | [3, | 22, 23]. | Con- |         |              |     |           |     |                  |
sequently,atretrievaltime,modelstendtolatchontodom- negation-aware baselines on MSR-VTT negation re-
|                  |       |          |            |                |             |         |        | trieval,   | matches | fine-tuned        | models | on COCO        | negated |
| ---------------- | ----- | -------- | ---------- | -------------- | ----------- | ------- | ------ | ---------- | ------- | ----------------- | ------ | -------------- | ------- |
| inant positive   | nouns | and      | attributes | while          | effectively |         | ignor- |            |         |                   |        |                |         |
|                  |       |          |            |                |             |         |        | retrieval, | and     | retains zero-shot |        | generalization | on non- |
| ing the negation |       | operator | [3,        | 22]. Empirical |             | studies | con-   |            |         |                   |        |                |         |
firm this: when queried with negated descriptions, strong negatedqueries,allwithoutmodifyingmodelparameters.
| VLMs frequently |          | return | images   | containing |      | the very | object |     |     |     |     |     |     |
| --------------- | -------- | ------ | -------- | ---------- | ---- | -------- | ------ | --- | --- | --- | --- | --- | --- |
| that was        | supposed | to be  | excluded | [3].       | This | failure  | is not |     |     |     |     |     |     |
subtle. AsillustratedinFigure1, giventhequery“Anim- 2.RelatedWork
ageofadognotonabeach,”CLIPretrievesbeachscenes
ofeveryrank[24].
2.1.Vision-LanguagePretrainingandCLIP
| Previous | works | [3, | 22] have | largely | addressed |     | nega- |     |     |     |     |     |     |
| -------- | ----- | --- | -------- | ------- | --------- | --- | ----- | --- | --- | --- | --- | --- | --- |
tionthroughadditionaltrainingorfine-tuningonsynthetic Large-scale contrastive vision-language pretraining has
datasets,wherecaptionsareprogrammaticallyeditedtoin- made CLIP-style models a standard backbone for image-
| troduce constructs |     | such | as “no | X,” “without |     | Y,” or | “A but |                |     |                     |     |           |                |
| ------------------ | --- | ---- | ------ | ------------ | --- | ------ | ------ | -------------- | --- | ------------------- | --- | --------- | -------------- |
|                    |     |      |        |              |     |        |        | text alignment |     | in tasks, including |     | retrieval | [24], caption- |
not B.” While such approaches improve performance on ing [16], depth estimation [13], and multimodal reason-
targetedbenchmarks[22],theyhavesignificantdrawbacks:
ing[2]. Thesemodelslearnajointembeddingspacefrom
highcomputationalcost,dependenceonlarge,curatedcor- internet-scaleimage-textpairs[2,24]andsupportzero-shot
pora,andoverfittingtothesyntheticpatternsobserveddur- transferviasimpletextprompts[24]. However,pretraining
| ing training. | As  | a result, | generalization |     | to  | natural | queries |     |     |     |     |     |     |
| ------------- | --- | --------- | -------------- | --- | --- | ------- | ------- | --- | --- | --- | --- | --- | --- |
corporaaretypicallyuncuratedanddonotcapturechalleng-
degrades,andnegationhandlingbecomesinseparablefrom inglinguisticphenomenasuchasnegationandcounterfac-
modelweights[7,25]. tuals[3,9].AssubsequentVLMscontinuetorelyonCLIP-
Evaluation progress has also been constrained by the likeencodersasscoringorguidancecomponents[16],their
scarcityofnegation-focusedbenchmarks,withearlyefforts inabilitytohandlenegationdescriptionsislargelyinherited
like MSR-VTT and COCO-based ones providing valuable fromtheoriginalcontrastivepretrainingobjective[3,24].
11354

2.2. Compositional and Training-Free Image-Text awarerepulsion,penalizingalignmentwithnegatedcontent
| Retrieval |     |     |     |     |     |     | withoutmodifyingthebasemodel’sweight. |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- |
Aparallellineofworkimprovescompositionalgeneraliza- 2.4.PositioningofOurApproach
tioninimage-textretrievalwithoutfine-tuningtheunderly-
ingencoder. CIReVLdemonstratesthattraining-freecom- Our work sits at the intersection of training-free composi-
|          |         |             |          |     |                  |     | tional retrieval |     | and negation-aware |     | vision-language. |     | Un- |
| -------- | ------- | ----------- | -------- | --- | ---------------- | --- | ---------------- | --- | ------------------ | --- | ---------------- | --- | --- |
| position | of text | embeddings, | combined |     | with lightweight |     |                  |     |                    |     |                  |     |     |
likefine-tuning-basedmethods,weintroducenonewlearn-
| re-ranking,     | cansignificantlyimproveretrievalforcomplex |               |          |                      |         |           |                 |        |        |           |                  |      |            |
| --------------- | ------------------------------------------ | ------------- | -------- | -------------------- | ------- | --------- | --------------- | ------ | ------ | --------- | ---------------- | ---- | ---------- |
|                 |                                            |               |          |                      |         |           | able parameters |        | and do | not       | require training | data | specific   |
| multi-attribute |                                            | queries [12]. | SpaceVLM |                      | extends | this idea |                 |        |        |           |                  |      |            |
|                 |                                            |               |          |                      |         |           | to negation.    | Unlike | prior  | geometric | methods,         |      | we explic- |
| by treating     | CLIP’s                                     | embedding     |          | space geometrically, |         | intro-    |                 |        |        |           |                  |      |            |
ducing angular composition operators that construct query itlymodeltheasymmetrybetweenaffirmativeandnegated
conceptsbyconditioningdynamicexclusionregionsonthe
| directions     | from       | multiple    | semantic | anchors       | while     | enforc- |                     |         |                                 |           |            |           |           |
| -------------- | ---------- | ----------- | -------- | ------------- | --------- | ------- | ------------------- | ------- | ------------------------------- | --------- | ---------- | --------- | --------- |
|                |            |             |          |               |           |         | LLM-predicted       |         | negation                        | strength. | Unlike     | heuristic | plug-     |
| ing separation |            | constraints | [25].    | These         | methods   | demon-  |                     |         |                                 |           |            |           |           |
|                |            |             |          |               |           |         | and-playapproaches, |         | weapplyaprincipledimage-sidere- |           |            |           |           |
| strate that    | meaningful | gains       | in       | compositional | retrieval | can     |                     |         |                                 |           |            |           |           |
|                |            |             |          |               |           |         | pulsion             | penalty | that preserves                  |           | the frozen | CLIP      | encoder’s |
beachievedusinggeometricoperationsonfrozenCLIPem-
|                       |     |     |                                |     |     |     | zero-shotgeneralization. |     |     | Together,thesecomponentsform |     |     |     |
| --------------------- | --- | --- | ------------------------------ | --- | --- | --- | ------------------------ | --- | --- | ---------------------------- | --- | --- | --- |
| beddingsalone[12,25]. |     |     | However,neitherapproachexplic- |     |     |     |                          |     |     |                              |     |     |     |
aunifiedtest-timescoringframeworkthatnarrowsthegap
itlymodelsthesemanticsofnegationortheasymmetrybe-
betweenfullytrainednegation-awaremodelsandpurelyge-
tweenconceptsthatarepresentandthosethatareexplicitly
ometrictraining-freebaselines.
absentinascene—agapourworkdirectlyaddresses.
| 2.3.Negation-AwareVision-LanguageModels |     |     |     |     |     |     | 3.Methodology |     |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
Recent work has shown that standard VLMs, including Our method is based on the geometric subspace model-
| CLIP [24], | often | fail in | captions | containing | explicit | nega- |        |          |      |     |                        |     |       |
| ---------- | ----- | ------- | -------- | ---------- | -------- | ----- | ------ | -------- | ---- | --- | ---------------------- | --- | ----- |
|            |       |         |          |            |          |       | ing of | SpaceVLM | [25] | and | the negation-sensitive |     | scor- |
tion, such as “no dog on the couch” or description of ab- ing of CLIPGLASSES [29]. We propose SpaceVLM-DRC
sentobjects[3,11,26,32]. Thisfailurestemsfromanaf- (Dynamic Repulsion with Context anchoring), a training-
firmative bias where models behave like “bags of words,” free inference-time framework for negation-aware image
prioritizingnounswhileeffectivelyignoringnegationoper- retrieval. It extends CLIP-based similarity scoring with-
| ators [3, | 11, 26]. | Benchmarks | such | as CC-Neg |     | [26], Neg- |                                           |     |     |     |     |     |         |
| --------- | -------- | ---------- | ---- | --------- | --- | ---------- | ----------------------------------------- | --- | --- | --- | --- | --- | ------- |
|           |          |            |      |           |     |            | outmodifyinganypretrainedmodelparameters. |     |     |     |     |     | Asshown |
Bench[3],andNegRefCOCOg[22]havebeendevelopedto in Fig. 2, an input caption is simultaneously processed by
expose these vulnerabilities, revealing that CLIP’s perfor- two LLM modules-a decomposer and a negation-strength
mancedropssharplywhendistinguishing“X”from“with- estimator-whose outputs are combined with CLIP encod-
outX”. Model-sideapproachessuchasCon-CLIP[26]and ings to construct negation-aware query directions. These
NegationCLIP[22]incorporatenegation-awaretrainingob-
|     |     |     |     |     |     |     | directions | are matched |     | against | image | embeddings | via co- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | ------- | ----- | ---------- | ------- |
jectivesorutilizeLLM-generatedsyntheticdatatoimprove sinesimilarity,andanimage-spacerepulsionpenaltyissub-
the recognition of absent concepts, although often at the tractedtoyieldthefinalscore.
costofadditionalfine-tuning.
3.1.EmbeddingSpaceandBaseRetrieval
| In specialized                                     |     | domains | such | as medical | imaging | [14], |          |     |                          |     |     |     |       |
| -------------------------------------------------- | --- | ------- | ---- | ---------- | ------- | ----- | -------- | --- | ------------------------ | --- | --- | --- | ----- |
| pairedimage-textdataandtechniquessuchasdynamicsoft |     |         |      |            |         |       |          | }N  |                          |     |     |     | }M    |
|                                                    |     |         |      |            |         |       | LetI ={I | j   | denotetheimagecorpusandC |     |     |     | ={c i |
|                                                    |     |         |      |            |         |       |          | j=1 |                          |     |     |     | i=1   |
labels or graph embeddings have been used to train mod- thesetoftextualcaptions.WeuseafrozenCLIPmodel[24]
els to distinguish normal findings from negative patholo- with image and text encoders (Φ ,Φ ) to project both
|     |     |     |     |     |     |     |     |     |     |     | img | txt |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
gies. More recently, Vu and Sheshappanavar [27] showed modalitiesintoasharedD-dimensionalℓ -normalizedem-
2
| that contrastive |          | fine-tuning   | of  | the text encoder   |     | alone can | beddingspace: |     |     |     |     |     |     |
| ---------------- | -------- | ------------- | --- | ------------------ | --- | --------- | ------------- | --- | --- | --- | --- | --- | --- |
| improve          | negation | understanding |     | in vision-language |     | mod-      |               |     |     |     |     |     |     |
els,increasingnegationretrievalaccuracybyupto15%in
|     |     |     |     |     |     |     |     | Φ   | (I  | )   |     | Φ   | (c ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
chestradiographdata. Althoughmanymethodsrequireac- img j txt i
|     |     |     |     |     |     |     | e I (I | j )= |       | ,   | e T (c i )= |     | (1)   |
| --- | --- | --- | --- | --- | --- | --- | ------ | ---- | ----- | --- | ----------- | --- | ----- |
|     |     |     |     |     |     |     |        | ∥Φ   | (I )∥ |     |             | ∥Φ  | (c )∥ |
cesstotrainingdataandgradients,neweralternativesoffer img j 2 txt i 2
| more flexible                                     | solutions: |     | SpaceVLM | [25] | models | negation |            |            |        |            |                |      |              |
| ------------------------------------------------- | ---------- | --- | -------- | ---- | ------ | -------- | ---------- | ---------- | ------ | ---------- | -------------- | ---- | ------------ |
|                                                   |            |     |          |      |        |          | Since both | embeddings |        | are        | ℓ -normalized, | as   | shown in     |
| asasemanticsubspaceratherthanasinglepoint,NEAT[7] |            |     |          |      |        |          |            |            |        |            | 2              |      |              |
|                                                   |            |     |          |      |        |          | equation   | 1, the     | cosine | similarity | reduces        | to a | dot product, |
employsparameter-efficienttest-timeadaptationofnormal-
givingthestandardCLIPretrievalscore:
izationlayers,andNEGTOME[11]addressesthestructural
| lossofnegationcuesthroughsemantictokenmerging. |     |     |     |     |     | Fi- |     |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
)⊤e
|                    |     |              |     |                 |     |       |     | S   | CLIP (I j | ,c i )=e | I (I j | T (c i ) | (2) |
| ------------------ | --- | ------------ | --- | --------------- | --- | ----- | --- | --- | --------- | -------- | ------ | -------- | --- |
| nally, CLIPGLASSES |     | [29]provides |     | anon-intrusive, |     | plug- |     |     |           |          |        |          |     |
and-playframeworkthatusesaLensmoduletodisentangle For captions in which the LLM detects no negation, S
CLIP
negated semantics and a Frame module to apply context- isuseddirectlyasthefinalscore,andtheremainderofthe
11355

Figure2.ArchitectureofSpaceVLM-DRC.AninputcaptionisprocessedbyaparallelLLMdecomposerandnegation-strengthestimator;
theiroutputsareencodedbyCLIPandcomposedintonegation-awarequerydirectionsviaSpaceVLMangulardecomposition. Cosine
similaritywithimageembeddingsandanimage-spacerepulsionpenaltyarecombinedtoproducethefinalretrievalscore.
| pipeline is | skipped. | Only captions | with a | confirmed | nega- |     |     |     |     |     |     |     |     |
| ----------- | -------- | ------------- | ------ | --------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
tionstructureproceedthroughthestagesbelow.
aff=“apicturewithagirl”
| 3.2. LLM-Based |     | Query | Decomposition |     | and |     |     | neg=“dog” |     |     |     |     |     |
| -------------- | --- | ----- | ------------- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
Negation-StrengthEstimation
cf=“apicturewithagirlandadog”
| Negation         | tokens    | such as      | “no”, “not”,   | and         | “with-  |           |         |           |            |             |             |              |      |
| ---------------- | --------- | ------------ | -------------- | ----------- | ------- | --------- | ------- | --------- | ---------- | ----------- | ----------- | ------------ | ---- |
|                  |           |              |                |             |         | If        | the LLM | returns   | a plain    | string      | rather than | a structured |      |
| out” are         | routinely | underweighed | by contrastive |             | vision- |           |         |           |            |             |             |              |      |
|                  |           |              |                |             |         | triplet   | (aff,   | neg,      | cf) and    | the caption | proceeds    | through      | the  |
| language         | models.   | To expose    | this structure |             | explic- |           |         |           |            |             |             |              |      |
|                  |           |              |                |             |         | remaining |         | pipeline. | Otherwise, |             | the caption | is scored    | with |
| itly, we process |           | each caption | c with two     | independent |         |           |         |           |            |             |             |              |      |
S directlyasdefinedinequation2.
CLIP
| prompts issued | to  | the frozen | instruction-tuned |     | model |                   |     |     |             |     |            |        |       |
| -------------- | --- | ---------- | ----------------- | --- | ----- | ----------------- | --- | --- | ----------- | --- | ---------- | ------ | ----- |
|                |     |            |                   |     |       | Negation-Strength |     |     | Estimation: |     | The second | prompt | clas- |
Qwen2.5-14B-Instruct[31].
sifiesthelinguisticstrengthofthenegation:
QueryDecomposition:Thefirstpromptdecomposescinto
astructuredtriplet:
|     |     |            |       |     |     |          |        | σ(c)∈{strong, |          | moderate,     | weak}     |             | (4)    |
| --- | --- | ---------- | ----- | --- | --- | -------- | ------ | ------------- | -------- | ------------- | --------- | ----------- | ------ |
|     |     |            |       |     |     | where    | strong | covers        | explicit | markers       | (“no,”    | “without”), |        |
|     |     | (aff, neg, | cf)=c |     | (3) |          |        |               |          |               |           |             |        |
|     |     |            |       |     |     | moderate |        | covers        | hedged   | constructions | (“doesn’t |             | have,” |
“lacks”),weakcoversimplicitabsences(“appearsabsent”).
where aff is a short phrase for affirmative that describes Eachoftheseclassesisassignedacosinethresholdtand
whatispresentinthescene,negisnegationortheexplicitly abaserepulsionweightω :
base
absentconcept,andcfisacounterfactualphrasedescribing
| thesceneasifthenegationswereremoved(i.e. |     |     |     | affandneg |     |     |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|                |                |     |     |     |     |     |     |      | (0.90, | 0.30) | σ =strong   |     |     |
| -------------- | -------------- | --- | --- | --- | --- | --- | --- | ---- | ------ | ----- | ----------- | --- | --- |
| co-occurring). | Forthecaption: |     |     |     |     |     |     |      |      |       |             |     |     |
|                |                |     |     |     |     |     | (t, | ω )= | (0.92, | 0.20) | σ =moderate |     | (5) |
base
(0.94,
|     |     |     |     |     |     |     |     |     |     | 0.10) | σ =weak |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------- | --- | --- |
“Apicturewithagirlbutnodog”
Alowertwidenstheangularexclusionzone(describedin
Thedecompositionyields: SpaceVLM[25]),applyinggreatergeometricpressureaway
11356

from the negated concept. The per-sample effective repul- 3.5. Image Encoding, Cosine Similarity Matching,
sionweightisω =ω ×ρ,whereρisarepulsionscale andFinalScoring
rep base
hyperparameter (default ρ = 1.5). This dynamic mapping
Image Encoding: In parallel with the text processing
extendsthefixedthresholdofSpaceVLM[25]andapprox-
pipeline, candidate images from the retrieval set are en-
imatesthetrainedadaptiveλofCLIPGLASSES[29]with-
codedoncebythefrozenCLIPimageencoderandℓ nor-
2
outanyfine-tuning.
malizedtoproduceimageembeddings{e (I )}.
I j
3.3.CLIPTextEncoding Cosine Similarity Matching: Each image embedding is
matched against the three composed query directions and
Oncedecompositionisconfirmed,allfourtextcomponents
againste :
neg
— aff, neg, cf, and the original caption c — are prefixed
with“Aphotoof”andindependentlyencodedbythefrozen s (j)=e (I )⊤dˆ , k ∈{1, 2, ctx} (10)
k I j k
CLIP text encoder (from equation 1) to yield four unit-
where s (j) measures how well the image I aligns with
normalizedembeddings(oneforeachcomponent): k j
eachcomposedquerydirectionfromEquation9. Inparal-
lel,theimageisalsomatcheddirectlyagainstthenegation
e ,e ,e ,e =e (aff,neg,cf,c) (6)
aff neg cf full T
conceptembedding:
Here,e encodesthecompleteoriginalcaptionandserves
full s (j)=e (I )⊤e (11)
asacontextanchorthatretainsricherscenesemanticsbe- neg I j neg
yondtheshortdecomposedphrase.Thesefourembeddings, wheres (j)measuresthevisualsimilaritybetweenimage
neg
togetherwiththethresholdtfromthenegation-strengthes- I and the negated concept. A high s (j) indicates the
j neg
timator,arepassedtotheangulardecompositionmodule. image visually contains the absent concept and should be
penalized.
3.4.AngularDecomposition(SpaceVLM)
RepulsionScore: Whiletheangulardirectionsreshapethe
Toconstructnegation-awarequerydirections,weapplythe queryintextspace,s fromequation11,directlymeasures
neg
angularcompositionoperatorofSpaceVLM[25]. Fortwo whether a candidate image visually resembles the negated
unit vectors a and b and the target cosine threshold t, we concept. Images with high s should be suppressed re-
neg
define: gardless of their alignment with the composed directions.
Wethereforedefinetherepulsionpenaltyasfollows:
α=arccos(t), θ =arccos (cid:0) a⊤b (cid:1) (7)
Repulsion(j)=−ω ·max(s (j), 0) (12)
where α is the angular exclusion radius determined by the rep neg
negation strength threshold t, and θ is the angle between whereω rep =ω base ×ρistheeffectiverepulsionweight,de-
unitvectors. Thecomposeddirectionis: terminedbythenegationstrengthclassσandtherepulsion
scale hyperparameter ρ. The max(·,0) term ensures that
sin
(cid:0)
α+
θ(cid:1)
sin
(cid:0)
α−
θ(cid:1)
onlyimageswithpositivesimilaritytothenegatedconcept
dˆ(a,b,t)= 2 a+ 2 b (8)
sinθ sinθ arepenalized.
Final Score: The sum of weighted directional scores rep-
Thesecondcoefficientisnaturallynegativewhenα<θ/2,
resenting the angular alignment and the repulsion penalty
geometrically repelling the composed vector away from b.
representing the image-space penalty are combined in the
The result is ℓ normalized. Using the four CLIP embed-
2
summationblockasshowninequation13:
dings and threshold t, three complementary query direc-
tionsarederived:
S (I ,c)=w s (j)+w s (j)+w s (j)
DRC j 1 1 2 2 ctx ctx
dˆ =dˆ(e ,e ,t) (cid:124) (cid:123)(cid:122) (cid:125)
1 aff neg angularalignment
(13)
dˆ 2 =dˆ(e aff ,e cf ,t) (9) + Repulsion(j)
dˆ =dˆ(e ,e ,t) (cid:124) (cid:123)(cid:122) (cid:125)
ctx full neg image-spacepenalty
dˆ isthecorequerydirection,pulledtowardtheaffirmative where
1
concept and pushed away from the negated one. dˆ adds
2
(w , w , w )=(0.35, 0.35, 0.30) (14)
a contrastive signal by repelling the query from the coun- 1 2 ctx
terfactual,penalizingsceneswherebothconceptsco-occur. The angular alignment term governs query geometry – di-
dˆ uses the full caption as the affirmative anchor instead recting the search toward the intended scene – while the
ctx
of the short decomposed phrase, preserving richer seman- repulsiontermgovernsimagegeometry–suppressingcan-
tics—a strategy inspired by the lens component of CLIP- didatesthatvisuallycontaintheabsentconcept.Theimages
GLASSES[29]. arerankedindescendingorderoftheS score.
DRC
11357

3.6.InferencePipeline which utilizes a Lens module for semantic disentangle-
mentandaFramemoduleforcontext-awarerepulsion.
| Our SpaceVLM-DRC         |     |     | is training-free,        | and | all components |     |          |           |          |     |       |      |           |
| ------------------------ | --- | --- | ------------------------ | --- | -------------- | --- | -------- | --------- | -------- | --- | ----- | ---- | --------- |
|                          |     |     |                          |     |                |     | Metrics: | We report | Recall@K |     | (R@1, | R@5, | R@10) and |
| remainfrozenatinference. |     |     | Ourthree-steppipelineis: |     |                |     |          |           |          |     |       |      |           |
medianrank(medR)metricsforCOCORetrieval-Neg,and
• Imagefeatureextraction: TheCLIPimageembeddings R@1, R@5, R@10 metrics for MSRVTT Retrieval-Neg.
| arecomputedfrom1,andℓ |     |     |     | normalizedforallimagesin |     |     |               |     |        |                 |     |        |                |
| --------------------- | --- | --- | --- | ------------------------ | --- | --- | ------------- | --- | ------ | --------------- | --- | ------ | -------------- |
|                       |     |     | 2   |                          |     |     | For Recall@K, |     | higher | values indicate |     | better | retrieval per- |
thecorpusonce,priortoanyqueryprocessing. formance(↑);formedR,lowervaluesarebetter(↓).
| • Caption | processing: |     | Each | caption | is simultaneously |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ---- | ------- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
passed to the LLM Decomposer and the LLM Negation 4.1.ImplementationDetails
| Strength                  | estimator. |      | If no negation               | is                   | detected, | the cap- |                                                      |     |       |                 |     |      |           |
| ------------------------- | ---------- | ---- | ---------------------------- | -------------------- | --------- | -------- | ---------------------------------------------------- | --- | ----- | --------------- | --- | ---- | --------- |
|                           |            |      |                              |                      |           |          | All experiments                                      |     | use a | frozen ViT-B/32 |     | CLIP | backbone. |
| tionisscoredwithstandardS |            |      |                              | .Otherwise,thedecom- |           |          |                                                      |     |       |                 |     |      |           |
|                           |            |      |                              | CLIP                 |           |          | Captiondecompositionandnegationstrengthestimationare |     |       |                 |     |      |           |
| posedtriplet(aff,         |            | neg, | cf)andtheoriginalcaptioncare |                      |           |          |                                                      |     |       |                 |     |      |           |
performedusingQwen2.5-14B-Instruct[31],which
| encodedbytheCLIPtextencoderasinequation6. |     |     |     |     |     | The    |                   |                    |                  |          |         |                 |              |
| ----------------------------------------- | --- | --- | --- | --- | --- | ------ | ----------------- | ------------------ | ---------------- | -------- | ------- | --------------- | ------------ |
|                                           |     |     |     |     |     |        | remains           | frozen throughout. |                  | Gradient | updates |                 | are not per- |
| dynamicthresholdtandtherepulsionweightω   |     |     |     |     |     | arere- |                   |                    |                  |          |         |                 |              |
|                                           |     |     |     |     |     | rep    | formedatanystage. |                    | Angularalignment |          |         | weightsaresetto |              |
trievedfromthestrengthmapping(fromequation5). (w ,w ,w ) = (0.35,0.35,0.30)withρ = 1.5asthede-
|                                   |     |          |     |              |             |     | 1 2        | ctx         |     |               |     |          |      |
| --------------------------------- | --- | -------- | --- | ------------ | ----------- | --- | ---------- | ----------- | --- | ------------- | --- | -------- | ---- |
| • Negation-aware                  |     | scoring: |     | The SpaceVLM | angular     | de- |            |             |     |               |     |          |      |
|                                   |     |          |     |              |             |     | fault. All | experiments |     | are conducted | on  | 8 NVIDIA | H100 |
| compositionmodulecomputesdˆ,dˆ,dˆ |     |          |     |              | fromtheCLIP |     |            |             |     |               |     |          |      |
1 2 ctx GPUs using data-parallel distributed evaluation, with im-
| text embeddings |     | and | the dynamic | threshold |     | t (equa- |     |     |     |     |     |     |     |
| --------------- | --- | --- | ----------- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
ageembeddingsbroadcastonceacrossallworkersandper-
| tion 9). | The | cosine | similarity | scores | s , s , | s (equa- |                                   |     |     |     |     |     |     |
| -------- | --- | ------ | ---------- | ------ | ------- | -------- | --------------------------------- | --- | --- | --- | --- | --- | --- |
|          |     |        |            |        | 1 2     | ctx      | captionmetricsaggregatedattheend. |     |     |     |     |     |     |
tion10), ands neg (equation11)arecomputedagainstall Although SpaceVLM-DRC does not require model
| image    | embeddings. |     | The repulsion | penalty  | is        | subtracted |           |                   |     |                     |     |       |            |
| -------- | ----------- | --- | ------------- | -------- | --------- | ---------- | --------- | ----------------- | --- | ------------------- | --- | ----- | ---------- |
|          |             |     |               |          |           |            | training, | the two           | LLM | modules—the         |     | query | decomposer |
| from the | weighted    | sum | to            | obtain S | (equation | 13),       |           |                   |     |                     |     |       |            |
|          |             |     |               | DRC      |           |            | and the   | negation-strength |     | estimator—introduce |     |       | additional |
whichisthenusedtorankallcandidateimages.
|     |     |     |     |     |     |     | per-query | inference | latency | compared |      | to CLIP | [24] and   |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | ------- | -------- | ---- | ------- | ---------- |
|     |     |     |     |     |     |     | SpaceVLM  | [25],     | which   | employs  | only | a       | single LLM |
4.ExperimentalResults module for query decomposition. This overhead is in-
|                                      |     |           |     |                |     |        | herent in | the dual-module |        | pipeline          |     | and is | most pro-   |
| ------------------------------------ | --- | --------- | --- | -------------- | --- | ------ | --------- | --------------- | ------ | ----------------- | --- | ------ | ----------- |
| Datasets:                            | We  | evaluated | our | SpaceVLM-DRC   |     | on two |           |                 |        |                   |     |        |             |
|                                      |     |           |     |                |     |        | nounced   | when using      | larger | instruction-tuned |     |        | models such |
| negation-focusedretrievalbenchmarks. |     |           |     | COCORetrieval- |     |        |           |                 |        |                   |     |        |             |
Qwen2.5-14B-Instruct
|     |     |     |     |     |     |     | as  |     |     |     | [31]. | As demonstrated |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --------------- | --- |
Neg[3]isderivedfromtheMS-COCOvalidationset[19].
|              |     |          |            |          |          |      | in SpaceVLM | [25], | substituting |     | a lighter | model | such as |
| ------------ | --- | -------- | ---------- | -------- | -------- | ---- | ----------- | ----- | ------------ | --- | --------- | ----- | ------- |
| Its captions | are | modified | to include | explicit | negation | con- |             |       |              |     |           |       |         |
TinyLlama-1Bachievesafavorableaccuracy-latencytrade-
| straints of | the form | “A  | but not | B,” requiring | models | to re- |          |              |     |                  |     |      |             |
| ----------- | -------- | --- | ------- | ------------- | ------ | ------ | -------- | ------------ | --- | ---------------- | --- | ---- | ----------- |
|             |          |     |         |               |        |        | off, and | our pipeline | is  | fully compatible |     | with | such alter- |
trieveimagesthatsatisfytheaffirmativeconditionwhileex-
|                           |     |     |                        |                |     |          | natives. | Alternatively, | mergingthetwoLLMpromptsinto |     |         |          |          |
| ------------------------- | --- | --- | ---------------------- | -------------- | --- | -------- | -------- | -------------- | --------------------------- | --- | ------- | -------- | -------- |
| cludingthenegatedconcept. |     |     | MSRVTTRetrieval-Neg[3] |                |     |          |          |                |                             |     |         |          |          |
|                           |     |     |                        |                |     |          | a single | call is a      | straightforward             |     | path to | reducing | the gap. |
| is a negation-augmented   |     |     | variant                | of the MSR-VTT |     | text-to- |          |                |                             |     |         |          |          |
Fortheretrievalsettingstargetedinthiswork(medicalim-
| video retrieval |         | benchmark  | [30],            | where queries   | are | negated   |                                          |             |         |              |               |         |             |
| --------------- | ------- | ---------- | ---------------- | --------------- | --- | --------- | ---------------------------------------- | ----------- | ------- | ------------ | ------------- | ------- | ----------- |
|                 |         |            |                  |                 |     |           | age search                               | and content |         | moderation), | batch         | offline | process-    |
| descriptions    | of      | video      | content.         | Both benchmarks |     | assess    |                                          |             |         |              |               |         |             |
|                 |         |            |                  |                 |     |           | ingmakesthislatencyacceptableinpractice. |             |         |              |               |         | Reducingin- |
| whether         | a model | can handle | presence-absence |                 |     | asymmetry |                                          |             |         |              |               |         |             |
|                 |         |            |                  |                 |     |           | ference                                  | overhead    | through | lighter      | decomposition |         | or unified  |
innatural-languagequeries.
promptingremainsapracticaldirectionforfuturework.
| Baselines: | WecompareSpaceVLM-DRCwiththefollow- |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ingbaselines:
|     |     |     |     |     |     |     | Table 1. | COCO Retrieval-Neg |     | Performance. |     | For | R@K, higher |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------------ | --- | ------------ | --- | --- | ----------- |
• CLIP Baseline [24]: Standard frozen CLIP with co- values(↑)andformedR,lowervalues(↓)arebetter.Bestresultin
sinesimilarityscoring,representingtheunmodifiedcon- bold;parenthesesshowimprovementsovertheCLIPbaseline.
trastivebaseline,whichoftenexhibitsanaffirmativebias.
• NEAT [7]: A negation-aware test-time adaptation Method R@1↑ R@5↑ R@10↑ medR↓
methodthatefficientlyadjustsdistribution-relatedparam-
|     |     |     |     |     |     |     | CLIPBaseline[24] |     |     | 25.0 | 47.9 | 59.1 | 6.0 |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | ---- | ---- | ---- | --- |
eters(specificallynormalizationlayers)duringinference CLIP+NEAT[7] 30.0(↑5.0) 54.6(↑6.7) 65.6(↑6.5) –
totackledual-conceptshifts. CLIP+SpaceVLM[25] 29.9(↑4.3) 55.1(↑7.2) 66.4(↑7.3) 4.0
• SpaceVLM [25]: A training-free geometric framework CLIP+Ours 30.2(↑5.2) 54.9(↑7.0) 65.5(↑6.4) 4.0
thatmodelsnegationasasemanticsubspaceratherthana
single point, deriving a representative direction from in- COCO Retrieval-Neg: Table 1 reports the results of the
tersectingconceptregionsinafrozenCLIPencoder. COCO negation retrieval benchmark. The CLIP baseline
• CLIP + SpaceVLM-DRC (Ours): Our methodology achievesanR@1of25.0,confirmingthewell-documented
incorporates the subspace modeling of SpaceVLM [25] inability of standard contrastive models to handle negated
andthedual-modulearchitectureofCLIPGLASSES[29], queries. NEAT, a test-time adaptation method that adjusts
11358

Figure3. Qualitativeretrievalresultsonastrong-negationcaption. Thequerycaptionreads: “Asamanexitsthebuildingtogreet
someone,thereisnotablynodiningtableinthissetting.”Thegroundtruthimage(blueborder,left)depictsamanwalkingoutsideabuilding
entrance.Row1–CLIPBaseline(redborders):CLIPretrievesindoorscenesprominentlyfeaturingdiningtables,failingtosuppressthe
negatedconcept. Row2–SpaceVLM-DRC(greenborders): SpaceVLM-DRCsuccessfullyretrievesthegroundtruthatRank1,with
theremainingresultsdepictingoutdoorbuildingandentrancescenes,confirmingcorrectnegationhandling. Rank3(noborder)isshared
bybothmethods,theonlyresultinwhichbothmodelsagree. Overall,SpaceVLM-DRCdemonstratesaclearimprovementoverCLIPby
replacingsemanticallyincorrectretrievalswithcontextuallyappropriateresults.
| Table2. MSRVTTRetrieval-NegPerformance. |                  |         |              | ForR@K,higher |      |                        |                                     |                        |     |     |     |
| --------------------------------------- | ---------------- | ------- | ------------ | ------------- | ---- | ---------------------- | ----------------------------------- | ---------------------- | --- | --- | --- |
|                                         |                  |         |              |               |      | to26.1(+2.3).          | SpaceVLM-DRCachievesthestrongestre- |                        |     |     |     |
| values (↑)                              | are better. Best | results | are in bold; | parentheses   | show |                        |                                     |                        |     |     |     |
|                                         |                  |         |              |               |      | sultsacrossallmetrics, |                                     | withanR@1of28.9(+5.1), |     |     | R@5 |
improvementsovertheCLIPbaseline. of 53.0 (+7.1), and R@10 of 63.2 (+6.6)— surpassing all
baselines,includingthetest-timeadaptationmethodNEAT,
|        |     | R@1↑ | R@5↑ |     | R@10↑ |                                                     |     |     |     |     |     |
| ------ | --- | ---- | ---- | --- | ----- | --------------------------------------------------- | --- | --- | --- | --- | --- |
| Method |     |      |      |     |       | anddoingsoentirelythroughinference-timegeometricand |     |     |     |     |     |
linguisticoperationsonafrozenCLIPencoder.
| CLIPBaseline[24] |     |     | 23.8 | 45.9 | 56.6 |     |     |     |     |     |     |
| ---------------- | --- | --- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- |
CLIP+NEAT[7] 24.8(↑1.0) 47.6(↑1.7) 58.1(↑1.5) Summary: Across both benchmarks, SpaceVLM-DRC
| CLIP+SpaceVLM[25] |     | 26.1(↑2.3) | 49.4(↑3.5) |     | 63.1(↑6.5) |     |     |     |     |     |     |
| ----------------- | --- | ---------- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
consistentlyoutperformsallbaselinesfortraining-freeand
| CLIP+Ours                               |                      | 28.9(↑5.1) | 53.0(↑7.1)   |           | 63.2(↑6.6) |                           |            |                                   |                           |                |           |
| --------------------------------------- | -------------------- | ---------- | ------------ | --------- | ---------- | ------------------------- | ---------- | --------------------------------- | ------------------------- | -------------- | --------- |
|                                         |                      |            |              |           |            | test-timeadaptation.      |            | Thegainsareparticularlypronounced |                           |                |           |
|                                         |                      |            |              |           |            | on MSRVTT,                | suggesting | that                              | our dynamic               | repulsion      | with      |
|                                         |                      |            |              |           |            | context-anchored          | mechanisms |                                   | generalizes               | well           | to video- |
| model behavior                          | at inference         |            | time without | modifying | pre-       |                           |            |                                   |                           |                |           |
|                                         |                      |            |              |           |            | languagenegationsettings. |            |                                   | Inparticular,SpaceVLM-DRC |                |           |
| trainedweights,improvesR@1to30.0(+5.0). |                      |            |              |           | SpaceVLM,  |                           |            |                                   |                           |                |           |
|                                         |                      |            |              |           |            | achieves these            | results    | without                           | training                  | data, gradient | up-       |
| operating                               | purely geometrically |            | on frozen    | CLIP      | embed-     |                           |            |                                   |                           |                |           |
dates,test-timeadaptation,orsyntheticcorpusconstruction.
| dings, reaches | an R@1 | of 29.9 | (+4.3), | demonstrating | that |     |     |     |     |     |     |
| -------------- | ------ | ------- | ------- | ------------- | ---- | --- | --- | --- | --- | --- | --- |
thetraining-freegeometriccompositioniscompetitivewith
4.2.QualitativeAnalysis
| adaptation-based | approaches. |     | Our method, |     | SpaceVLM- |     |     |     |     |     |     |
| ---------------- | ----------- | --- | ----------- | --- | --------- | --- | --- | --- | --- | --- | --- |
DRC, achieves an R@1 of 30.2 (+5.2), matching NEAT’s Figure 3presents aqualitative retrievalexample fora cap-
performancewhilealsoattainingamedianrankof4.0—on tioncontainingstrongnegation.TheCLIPbaselinedoesnot
parwiththebest-performingbaselines—anddoingsowith- retrievethecorrectimagewithinthetop5;instead,itreturns
outanyparameterupdatesortest-timeadaptationoverhead. indoorscenes(exceptRank3)thatprominentlyfeaturedin-
MSRVTTRetrieval-Neg:Table2reportstheresultsofthe ingtables,whichisnegated. Incontrast,SpaceVLM-DRC
MSRVTT negation retrieval benchmark. The CLIP base- successfully retrieves the ground-truth image at Rank 1,
linescoresanR@1of23.8,withNEATprovidingamodest withallfiveresultsdepictingoutdoorentrancesettingscon-
improvement to 24.8 (+1.0). SpaceVLM improves further sistentwiththecaption’saffirmativecontent. Inparticular,
11359

the Rank 3 result is shared by both methods, representing ρ = 0.0 completely disables the repulsion penalty and re-
theonlypointofagreementbetweenCLIPandSpaceVLM- ducesthethreeangularalignmentscores.
DRC. This improvement is attributed to the interplay of
three components: (i) the context anchor, which uses the 5.ConclusionandFutureWork
full caption embedding as an affirmative anchor to cap-
Inthispaper,wepresentSpaceVLM-DRC,atraining-free,
ture richer scene-level semantics; (ii) the dynamic thresh-
negation-aware retrieval framework that operates entirely
old, which calibrates the exclusion zone according to the
at inference time on a frozen CLIP backbone. Our novel
detectedstrong-negationstrength,and(iii)thedirectrepul-
method combines LLM-based query decomposition, dy-
sion term, which further penalizes images whose embed-
namiccontrolofnegationstrength,andageometricimage-
dingsalignwiththenegatedconcept.
space repulsion mechanism to reshape similarity scores
without gradient updates or synthetic corpus construction.
Table3. Component-wiseablationonCOCORetrieval-Neg. One
Empirically, this design narrows the gap to fine-tuned
novelcomponentisaddedtoeachrow.Bestresultsinbold.
negation-aware models on COCO Retrieval-Neg and out-
performs both geometric and test-time adaptation base-
Method R@1↑ R@5↑ R@10↑ medR↓
lines on MSRVTT Retrieval-Neg, while preserving zero-
CLIPBaseline 25.0 47.9 59.1 6.0
shotperformanceonnon-negatedqueries—showingthatro-
+dˆ only 27.2 50.7 61.7 5.0
1 bust negation handling can be achieved without retraining
+dˆ +dˆ 27.5 51.1 62.1 5.0
1 2 theunderlyingvision–languageencoder.
+dˆ 1 +dˆ 2 +dˆ ctx 29.1 53.2 64.3 5.0 Future work will extend this inference-time framework
+dˆ 1 +dˆ 2 +dˆ ctx +Rep. 30.2 54.9 65.5 4.0 to stronger VLM backbones and video–text encoders. We
alsoplantoincorporateadditionallinguisticoperators,such
as conjunctions, disjunctions, and quantifiers, alongside
negation. Ontheefficiencyside,weaimtoreducetheper-
4.3.AblationStudy
querylatencyintroducedbythedualLLMmodules—either
WeconductedtwoablationstudiesontheCOCORetrieval- by unifying the decomposer and negation-strength es-
Neg benchmark to analyze the contribution of each com- timator into a single prompt, or by replacing the larger
ponentofSpaceVLM-DRC:(1)acomponent-wiseablation Qwen2.5-14B-Instruct [31] model with a lighter
alternative such as TinyLlama-1B, which SpaceVLM [25]
that progressively adds each module and (2) a sweep over
has shown to offer a strong accuracy-latency tradeoff.
therepulsionscalehyperparameterρ.
Finally, future efforts will focus on developing larger and
morediversebenchmarksthatcaptureopen-world,long-tail
Table4. Ablationstudyontherepulsionscalehyperparameterρ
visualconceptsandmulti-sentenceinstructions.
onCOCORetrieval-Neg.Bestresultsinbold.
Method R@1↑ R@5↑ R@10↑ medR↓ References
ρ=0.0(norepulsion) 27.7 51.7 62.7 5.0
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
ρ=0.5 29.0 53.2 64.3 5.0
mad,IlgeAkkaya,FlorenciaLeoniAleman,DiogoAlmeida,
ρ=1.0 29.9 54.4 65.2 4.0
JankoAltenschmidt, SamAltman, ShyamalAnadkat, etal.
ρ=1.5(default) 30.2 54.9 65.5 4.0
Gpt-4 technical report. arXiv preprint arXiv:2303.08774,
ρ=2.0 29.7 54.4 65.3 4.0
2023. 2
[2] Jean-BaptisteAlayrac, JeffDonahue, PaulineLuc, Antoine
Table 3 reports the effect of progressively adding each Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Men-
component of SpaceVLM-DRC on top of the CLIP base- sch,KatherineMillican,MalcolmReynolds,etal.Flamingo:
line. StartingfromvanillaCLIP,wefirstintroducethestan- a visual language model for few-shot learning. Advances
dardSpaceVLMdirectiondˆ (affirmativevs.negated),then inneuralinformationprocessingsystems,35:23716–23736,
1
add the counterfactual direction dˆ (affirmative vs. coun- 2022. 2
2
terfactual), then add the context anchor direction dˆ (full [3] Kumail Alhamoud, Shaden Alshammari, Yonglong Tian,
ctx
GuohaoLi,PhilipHSTorr,YoonKim,andMarzyehGhas-
captionvs.negated),andfinallyaddtheimage-spacerepul-
semi. Vision-languagemodelsdonotunderstandnegation.
sionpenalty. Allconfigurationsusethedynamicthreshold
InCVPR,pages29612–29622,2025. 2,3,6
unlessotherwisestated.
[4] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur,
The effective repulsion weight per-sample is ω =
rep Daniel C Castro, Anton Schwaighofer, Stephanie Hyland,
ω base × ρ, where ω base is determined by the negation- Maria Wetscherek, Tristan Naumann, Aditya Nori, Javier
strength estimator, and ρ is a repulsion scale hyperparam- Alvarez-Valle, et al. Making the most of text semantics to
eter. Table 4 sweeps ρ over {0.0,0.5,0.1,1.5,2.0}, where improve biomedical vision–language processing. In Euro-
11360

peanconferenceoncomputervision,pages1–21.Springer, large vision language models: Benchmark evaluations and
2022. 2 challenges. InProceedingsoftheComputerVisionandPat-
[5] AbrahamMichaelFowler.Negationinnaturallanguagepro- ternRecognitionConference,pages1587–1606,2025. 2
cessing. TheUniversityofTexasatDallas,2006. 2 [19] Tsung-YiLin,MichaelMaire,SergeBelongie,JamesHays,
[6] Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng PietroPerona,DevaRamanan,PiotrDolla´r,andCLawrence
Liu, and Jianfeng Gao. Vision-language pre-training: Ba- Zitnick. Microsoft coco: Common objects in context. In
sics, recent advances, and future trends. arXiv preprint European conference on computer vision, pages 740–755.
arXiv:2210.09263,2022. 2 Springer,2014. 6
[7] Haochen Han, Alex Jinpeng Wang, Fangming Liu, and [20] HaotianLiu,ChunyuanLi,QingyangWu,andYongJaeLee.
Jun Zhu. Negation-aware test-time adaptation for vision- Visual instruction tuning. Advances in neural information
language models. arXiv preprint arXiv:2507.19064, 2025. processingsystems,36:34892–34916,2023. 2
2,3,6,7 [21] JiasenLu,DhruvBatra,DeviParikh,andStefanLee.Vilbert:
[8] IrynaHartsockandGhulamRasool. Vision-languagemod- Pretraining task-agnostic visiolinguistic representations for
elsformedicalreportgenerationandvisualquestionanswer- vision-and-languagetasks. Advancesinneuralinformation
ing:areview. FrontiersinArtificialIntelligence,Volume7- processingsystems,32,2019. 2
2024,2024. 2 [22] JunsungPark,JungbeomLee,JongyoonSong,SangwonYu,
[9] Cheng-YuHsieh,JieyuZhang,ZixianMa,AniruddhaKem- DahuinJung,andSungrohYoon.Know”no”better:Adata-
bhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable driven approach for enhancing negation awareness in clip.
benchmarksforvision-languagecompositionality.Advances In Proceedings of the IEEE/CVF International Conference
inneuralinformationprocessingsystems,36:31096–31116, onComputerVision,pages2825–2835,2025. 2,3
2023. 2 [23] VincentQuantmeyer,PabloMosteiro,andAlbertGatt. How
[10] ChaoJia,YinfeiYang,YeXia,Yi-TingChen,ZaranaParekh, andwheredoesclipprocessnegation? InProceedingsofthe
HieuPham, QuocLe, Yun-HsuanSung, ZhenLi, andTom 3rdWorkshoponAdvancesinLanguageandVisionResearch
Duerig.Scalingupvisualandvision-languagerepresentation (ALVR),pages59–72,2024. 2
learningwithnoisytextsupervision. InICML,pages4904– [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
4916.PMLR,2021. 2 Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
[11] InhaKang, YoungsunLim, SeonhoLee, JihoChoi, Junsuk Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Choe,andHyunjungShim. What”not”todetect:Negation- Krueger, and Ilya Sutskever. Learning transferable visual
aware vlms via structured reasoning and token merging. modelsfromnaturallanguagesupervision. InICML,pages
arXivpreprintarXiv:2510.13232,2025. 3 8748–8763.PMLR,2021. 1,2,3,6,7
[12] ShyamgopalKarthik,KarstenRoth,MassimilianoMancini, [25] Sepehr Kazemi Ranjbar, Kumail Alhamoud, and Marzyeh
and Zeynep Akata. Vision-by-language for training- Ghassemi. Spacevlm: Sub-space modeling of negation in
free compositional image retrieval. arXiv preprint vision-languagemodels. arXivpreprintarXiv:2511.12331,
arXiv:2310.09291,2023. 3 2025. 2,3,4,5,6,7,8
[13] Nischal Khanal and Shivanand Venkanna Sheshappanavar. [26] Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa
Edadepth:Enhanceddataaugmentationformonoculardepth Singh, and Aparna Bharati. Learn” no” to say” yes” bet-
estimation. In 2024 International Conference on Machine ter: Improvingvision-languagemodelsvianegations. arXiv
LearningandApplications(ICMLA),pages620–627.IEEE, preprintarXiv:2403.20312,2024. 3
2024. 2 [27] JasmineVuandShivanandVenkannaSheshappanavar. Im-
[14] HanbinKoandChang-MinPark. Bringingcliptotheclinic: provingnegationunderstandinginmedicalvision-language
Dynamicsoftlabelsandnegation-awarelearningformedical models via contrastive fine-tuning. In Proceedings of the
analysis.InProceedingsoftheComputerVisionandPattern IEEE/CVFWinterConferenceonApplicationsofComputer
RecognitionConference,pages25897–25906,2025. 3 Vision,pages395–404,2026. 3
[15] TonyLee,HaoqinTu,ChiHWong,WenhaoZheng,Yiyang [28] ZiyueWang,AozhuChen,FanHu,andXirongLi. Learnto
Zhou, Yifan Mai, Josselin S Roberts, Michihiro Yasunaga, understandnegationinvideoretrieval.InProceedingsofthe
HuaxiuYao,CihangXie,etal.Vhelm:Aholisticevaluation 30th ACM International Conference on Multimedia, pages
ofvisionlanguagemodels. AdvancesinNeuralInformation 434–443,2022. 2
ProcessingSystems,37:140632–140666,2024. 2 [29] JunhaoXiao,ZhiyuWu,HaoLin,YiChen,YahuiLiu,Xi-
[16] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. aoran Zhao, Zixu Wang, and Zejiang He. Not just what’s
Blip: Bootstrappinglanguage-imagepre-trainingforunified there: Enablingcliptocomprehendnegatedvisualdescrip-
vision-language understanding and generation. In ICML, tionswithoutfine-tuning. arXivpreprintarXiv:2602.21035,
pages12888–12900.PMLR,2022. 2 2026. 3,5,6
[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. [30] JunXu,TaoMei,TingYao,andYongRui. Msr-vtt:Alarge
Blip-2: Bootstrapping language-image pre-training with videodescriptiondatasetforbridgingvideoandlanguage.In
frozenimageencodersandlargelanguagemodels.InICML, ProceedingsoftheIEEEconferenceoncomputervisionand
pages19730–19742.PMLR,2023. 2 patternrecognition,pages5288–5296,2016. 6
[18] Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu, Huy [31] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen
Nghiem, and Guangyao Shi. A survey of state of the art Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng
11361

| Liu, Fei               | Huang, Guanting | Dong,     | Haoran         | Wei, Huan   | Lin,      |
| ---------------------- | --------------- | --------- | -------------- | ----------- | --------- |
| Jialong Tang,          | Jialin Wang,    | Jian      | Yang, Jianhong |             | Tu, Jian- |
| wei Zhang,             | Jianxin Ma,     | Jin Xu,   | Jingren        | Zhou, Jinze | Bai,      |
| Jinzheng               | He, Junyang     | Lin, Kai  | Dang, Keming   | Lu,         | Keqin     |
| Chen, Kexin            | Yang, Mei       | Li,       | Mingfeng Xue,  | Na          | Ni, Pei   |
| Zhang, Peng            | Wang, Ru        | Peng,     | Rui Men, Ruize | Gao,        | Runji     |
| Lin, Shijie            | Wang, Shuai     | Bai,      | Sinan Tan,     | Tianhang    | Zhu,      |
| Tianhao                | Li, Tianyu Liu, | Wenbin    | Ge, Xiaodong   | Deng,       | Xi-       |
| aohuan Zhou,           | Xingzhang       | Ren,      | Xinyu Zhang,   | Xipin       | Wei,      |
| Xuancheng              | Ren, Yang       | Fan, Yang | Yao, Yichang   | Zhang,      | Yu        |
| Wan, Yunfei            | Chu, Yuqiong    | Liu,      | Zeyu Cui,      | Zhenru      | Zhang,    |
| and Zhihao             | Fan. Qwen2      | technical | report.        | arXiv       | preprint  |
| arXiv:2407.10671,2024. |                 | 4,6,8     |                |             |           |
[32] YuhuiZhang,YuchangSu,YimingLiu,andSerenaYeung-
| Levy. Negvqa: | Can | vision | language models | understand |     |
| ------------- | --- | ------ | --------------- | ---------- | --- |
negation? InFindingsoftheAssociationforComputational
| Linguistics:ACL2025,pages3707–3716,2025. |     |     |     | 3   |     |
| ---------------------------------------- | --- | --- | --- | --- | --- |
11362
