Logical Consistency is Vital: Neural-Symbolic Information Retrieval for
|     |     |     | Negative-Constraint |     | Queries |     |     |     |     |
| --- | --- | --- | ------------------- | --- | ------- | --- | --- | --- | --- |
GanlinXu1♦* ZhoujiaZhang1♦* WangyiMei1♦ JiaqingLiang1 † WeijiaLu2▲
XiaodongZhang2▲ ZhifeiYang2▲ XiaofengMa2▲ YanghuaXiao3 DeqingYang1 †
1SchoolofDataScience,FudanUniversity,Shanghai,China
2UnitedAutomotiveElectronicSystems,Shanghai,China
3CollegeofComputerScienceandArtificialIntelligence,FudanUniversity,Shanghai,China
♦{glxu24,zhangzj24,wymei24}@m.fudan.edu.cn
{liangjiaqing,shawyh,yangdeqing}@fudan.edu.cn
▲{alfredwjlu,xiaodong.zhang.chn,yangzhifei006,maxf0124}@gmail.com
|     |     | Abstract |     |     | Query: In what city was Yao Ming born? |     |     |     |     |
| --- | --- | -------- | --- | --- | -------------------------------------- | --- | --- | --- | --- |
Negtive document:                           Similarity score:0.72
Ming Yao is a retired Chinese professional basketball
Informationretrievalplaysacrucialroleinre-
player who is widely regarded as one of the greatest
| source localization. |     | Current | dense | retrievers |     |     |     |     |     |
| -------------------- | --- | ------- | ----- | ---------- | --- | --- | --- | --- | --- |
basketball players from Chin...Ming Yao played as a center
retrievetherelevantdocumentswithinacorpus
and had a dominant presence on the court...Ming Yao
viaembeddingsimilarities,whichcomputesim- played for the Rockets for the entirety of his NBA career,
ilaritiesbetweendensevectorsmainlydepend- from 2002 to 2011. During his time in the NBA, Ming Yao
ing on word co-occurrence between queries was an 8-time NBA All-Star and was inducted into the
Naismith Memorial Basketball Hall of Fame in 2016.
anddocuments,butoverlooktherealqueryin-
Positive document:                           Similarity score:0.67
| tents. Thus,theyoftenretrievenumerousirrel- |     |     |     |     |           |                  |             |          |              |
| ------------------------------------------- | --- | --- | --- | --- | --------- | ---------------- | ----------- | -------- | ------------ |
|                                             |     |     |     |     | Shanghai  | is  a  bustling  | metropolis  | located  | on  China's  |
evantdocuments. Particularlyinthescenarios eastern coast, along the Yangtze River Delta... Shanghai
ofcomplexqueriessuchasnegative-constraint has  produced  many  famous  historical  and  modern
queries, their retrieval performance could be celebrities, such as Shi Hu, Ailing Zhang, Ming Yao.
Known for its impressive skyline dominated by modern
| catastrophic. | To  | address | the issue, | we pro- |     |     |     |     |     |
| ------------- | --- | ------- | ---------- | ------- | --- | --- | --- | --- | --- |
skyscrapers like the iconic Oriental Pearl Tower and the
| pose a neuro-symbolic |     | information |     | retrieval |     |     |     |     |     |
| --------------------- | --- | ----------- | --- | --------- | --- | --- | --- | --- | --- |
Shanghai Tower, the city is a striking blend of the old and
| method, | namely | NS-IR, | that leverages | first- | the new.  |     |     |     |     |
| ------- | ------ | ------ | -------------- | ------ | --------- | --- | --- | --- | --- |
orderlogic(FOL)tooptimizetheembeddings
of naive natural language by considering the Figure1: AnillustrationofBGE-basedretrieval. The
logicalconsistency betweenqueriesanddoc- word marked in green is the co-occurrence word be-
uments. Specifically,weintroducetwonovel tweenthequeryanddocuments.
techniques,logicalignmentandconnectivecon-
straint,torerankcandidatedocuments,thereby
| enhancingretrievalrelevance. |     |     | Furthermore,we |     |     |     |     |     |     |
| ---------------------------- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
searchengines(Lietal.,2022),questionanswering
constructanewdatasetNegConstraintinclud-
(Zhaoetal.,2021)andretrieval-augmentedgener-
ingnegative-constraintqueriestoevaluateour
ation(RAG)systems(Hulyetal.,2024),offering
NS-IR’sperformanceonsuchcomplexIRsce-
significantimprovementsinIR.
| narios. Ourextensiveexperimentsdemonstrate |     |     |     |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thatNS-IRnotonlyachievessuperiorzero-shot
Theembeddings(representations)generatedby
retrievalperformanceonwebsearchandlow- denseretrievers(suchasBGE(Xiaoetal.,2024))
resourceretrievaltasks,butalsoperformsbetter
|                        |     |          |     |             | focus on | overall | semantic similarity, | which | is ca- |
| ---------------------- | --- | -------- | --- | ----------- | -------- | ------- | -------------------- | ----- | ------ |
| on negative-constraint |     | queries. |     | Our scource |          |         |                      |       |        |
pableofhandlingsemanticallysimilarwordscom-
| code and | dataset | are available |     | at https:// |     |     |     |     |     |
| -------- | ------- | ------------- | --- | ----------- | --- | --- | --- | --- | --- |
paredtosparseretrieval(suchasBM25(Robertson
github.com/xgl-git/NS-IR-main.
|     |     |     |     |     | et al., 2009)) | that | uses keyword | matching. | How- |
| --- | --- | --- | --- | --- | -------------- | ---- | ------------ | --------- | ---- |
1 Introduction ever,denseretrievalstillreliesonsuperficialword
|     |     |     |     |     | co-occurrencebetweenqueriesanddocuments. |     |     |     | As  |
| --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- |
Information retrieval (IR) tasks aim at obtaining illustratedinFigure1,thenegativedocumenthas
relevantinformationfromlarge-scaledatacollec-
abiggerscorethanthepositivedocumentjustbe-
| tion,suchasdocumentsanddatabases. |     |     |     | Densere- |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
causethequery’skeyword“MingYao”occursin
trieval (Karpukhin et al., 2020a) is an advanced the former more frequently. Thus, the approach
informationretrievaltechniquefocusingonseman-
|               |              |         |     |               | fails to   | understand | the real  | query intent, | thereby    |
| ------------- | ------------ | ------- | --- | ------------- | ---------- | ---------- | --------- | ------------- | ---------- |
| tic embedding | similarities | between |     | texts. It has |            |            |           |               |            |
|               |              |         |     |               | retrieving | irrelevant | documents | (Wu et        | al., 2024; |
beenwidelyadoptedinmanyapplicationssuchas
Fangetal.,2024).
Notably,theretrievalapproachesbasedonword
*Equalcontribution.
†Co-correspondingauthors. co-occurrence have to face some challenges on
1828
FindingsoftheAssociationforComputationalLinguistics:ACL2025,pages1828–1847
July27-August1,2025©2025AssociationforComputationalLinguistics

|     |     |     |     |     |     |     | independent         |         | techniques: | 1)         | Logic | alignment:   | To   |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | ------- | ----------- | ---------- | ----- | ------------ | ---- |
|     |     |     |     |     |     |     | incorporate         | overall | logic       | semantics  |       | in FOL       | into |
|     |     |     |     |     |     |     | NL representations, |         |             | we measure |       | distribution | dif- |
ferencesbetweenNLandFOLembeddingsusing
optimaltransport(Redkoetal.,2019)andupdate
theembeddingsofqueriesanddocumentsrespec-
|     |     |     |     |     |     |     | tively. 2)Connectiveconstraint: |     |     |     | Toreflecttherole |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | ---------------- | --- | --- |
ofpartialwordsinFOLonlogicalconsistency,we
|     |     |     |     |     |     |     | leverage  | different | words     | in  | FOL        | (especially | con-  |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --------- | --- | ---------- | ----------- | ----- |
|     |     |     |     |     |     |     | nectives) | to render | different |     | attentions | to          | words |
inNL,generatingbetterembeddingswithlogical
|     |     |     |     |     |     |     | semantics. | We  | leverage | these | two | techniques | to  |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | -------- | ----- | --- | ---------- | --- |
recalculatesimilarityscoresandrerankcandidate
documents.
Figure2: AretrievalexampleofGooglesearchengine.
|     |     |     |     |     |     |     | To evaluate         |     | the   | performance |     | of NS-IR | in   |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ----- | ----------- | --- | -------- | ---- |
|     |     |     |     |     |     |     | negative-constraint |     | query | settings,   |     | we have  | con- |
complex queries, particularly involving negative- structed and released a dataset, namely NegCon-
| constraint   | queries, | due   | to  | the neglect | of      | logical |            |         |          |       |         |      |           |
| ------------ | -------- | ----- | --- | ----------- | ------- | ------- | ---------- | ------- | -------- | ----- | ------- | ---- | --------- |
|              |          |       |     |             |         |         | straint,   | which   | contains | three | types   | of   | negative- |
| consistency. | As       | shown | in  | Figure      | 21, for | a gen-  |            |         |          |       |         |      |           |
|              |          |       |     |             |         |         | constraint | queries | and      | was   | sourced | from | the       |
eral query “What are the RAG methods that do Wikipedia dump (Karpukhin et al., 2020a). The
notinvolvepromptengineering?”,thedocuments experiments show our NS-IR’s superiority over
| returned    | by a | keyword-based |         | search | engine  | of-   |                       |     |     |        |           |     |         |
| ----------- | ---- | ------------- | ------- | ------ | ------- | ----- | --------------------- | --- | --- | ------ | --------- | --- | ------- |
|             |      |               |         |        |         |       | some state-of-the-art |     |     | (SOTA) | baselines |     | on Neg- |
| ten contain | the  | excluded      | keyword |        | “prompt | engi- |                       |     |     |        |           |     |         |
Constraint,anditspotentialforhandlingcomplex
| neering”, | which    | are not    | logically | consistent    |     | with | logicalqueries. |     |     |     |     |     |     |
| --------- | -------- | ---------- | --------- | ------------- | --- | ---- | --------------- | --- | --- | --- | --- | --- | --- |
| the query | intent2. | Therefore, |           | understanding |     | real |                 |     |     |     |     |     |     |
Themaincontributionsofthispaperinclude:
queryintentsrequiresensuringlogicalconsistency
|     |     |     |     |     |     |     | 1. To | address | typical | complex |     | logical | queries |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------- | ------- | ------- | --- | ------- | ------- |
between queries and documents besides seman- inIR,i.e.,negative-constraintqueries,wepropose
tic similarity. First-order logic (FOL), as a for- NS-IR which combines the strengths of NL and
mallogicalsystem,offersclearlogicalsemantics
FOLtosynthesizesemanticsimilarityandlogical
| and expresses |     | complex | relations | in  | natural | lan- |     |     |     |     |     |     |     |
| ------------- | --- | ------- | --------- | --- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
consistency.
guage(Barwise,1977). Forinstance,theFOLof 2. Weintroducetwokeytechniques: logicalign-
| the aforementioned |     | query |     | is “RAGMethod(x) |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | ----- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ment(Sec.4.2)andconnectiveconstraint(Sec.4.3),
∧
| InvolvesPromptEngineering(x)”, |     |     |     |     | which | clearly |             |       |     |            |     |         |     |
| ------------------------------ | --- | --- | --- | --- | ----- | ------- | ----------- | ----- | --- | ---------- | --- | ------- | --- |
|                                |     |     |     |     |       |         | to optimize | naive | NL  | embeddings |     | by FOL, | and |
¬
expressesthenegativesemanticsthroughthelogi-
|                |     |     |     |     |     |     | rerank       | candidate | documents |     | to improve |     | retrieval |
| -------------- | --- | --- | --- | --- | --- | --- | ------------ | --------- | --------- | --- | ---------- | --- | --------- |
| calconnective‘ |     | ’.  |     |     |     |     | performance. |           |           |     |            |     |           |
¬
Inlightofthis,throughinvestigatingthepoten-
|     |     |     |     |     |     |     | 3. We | further | release | a   | new dataset |     | NegCon- |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------- | ------- | --- | ----------- | --- | ------- |
tialimpactofFOLoncomplexlogicalqueries,we
|     |     |     |     |     |     |     | straint(Sec.5), |     | whichcanbeutilizedasabench- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --------------------------- | --- | --- | --- | --- |
proposeaNeural-SymbolicInformationRetrieval mark for negative-constraint queries. Our exten-
method(NS-IR)torerankthecandidatedocuments
|                               |          |            |     |                    |         |     | sive experimental |     | results  | on       | public | datasets      | and |
| ----------------------------- | -------- | ---------- | --- | ------------------ | ------- | --- | ----------------- | --- | -------- | -------- | ------ | ------------- | --- |
| returned                      | by dense | retrievers |     | based              | on FOL, | for |                   |     |          |          |        |               |     |
|                               |          |            |     |                    |         |     | NegConstraint     |     | show     | that our | NS-IR  | significantly |     |
| moreaccurateretrievalresults. |          |            |     | Specifically,NS-IR |         |     |                   |     |          |          |        |               |     |
|                               |          |            |     |                    |         |     | outperforms       |     | the SOTA | methods  |        | on vanilla    | and |
firstretrievesasetoforderedcandidatedocuments
|              |       |            |         |         |         |      | negative-constraint |     | queries                       |     | in zero-shot |     | settings, |
| ------------ | ----- | ---------- | ------- | ------- | ------- | ---- | ------------------- | --- | ----------------------------- | --- | ------------ | --- | --------- |
| using a      | dense | retriever, | then    | employs | large   | lan- |                     |     |                               |     |              |     |           |
|              |       |            |         |         |         |      | respectively.       |     | Ourworkinthispaperpavestheway |     |              |     |           |
| guage models |       | (such as   | GPT-4o) | to      | convert | both |                     |     |                               |     |              |     |           |
forfutureresearchoncomplexqueriesinIR.
| the query | and | documents | into | FOL. | To  | incorpo- |     |     |     |     |     |     |     |
| --------- | --- | --------- | ---- | ---- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
ratelogicalconsistencyandoptimizeembeddings
|          |         |          |       |     |         |     | 2 RelatedWork      |     |     |     |     |     |     |
| -------- | ------- | -------- | ----- | --- | ------- | --- | ------------------ | --- | --- | --- | --- | --- | --- |
| of naive | natural | language | (NL), | we  | propose | two |                    |     |     |     |     |     |     |
|          |         |          |       |     |         |     | 2.1 DenseRetrieval |     |     |     |     |     |     |
1Intheexample,thesearchengine’sretrievalisbasedon
BM25algorithm,butdenseretrievalproducessimilarresults Denseretrievalhasgainedsignificantattentionin
innegative-constraintqueries.
|     |     |     |     |     |     |     | information | retrieval |     | due to | its advantages |     | over |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | --- | ------ | -------------- | --- | ---- |
2Althoughsomesearchtechniquesinsearchenginescan
|     |     |     |     |     |     |     | traditional | sparse | vector | space | models. |     | Sparse |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | ------ | ----- | ------- | --- | ------ |
use‘-’tofilterkeywords(suchas“WhataretheRAGmethods
-promptengineering”),theyarenotfamiliartoordinaryusers. modelsrepresentdocuments andqueriesashigh-
1829

dimensionalvectorswithmostlyzerovalues(Yang idze,2017). However,duetotheinherentcomplex-
et al., 2017; Chen et al., 2017). Dense retrieval ityofnaturallanguage,thesemethodsstruggleto
modelsencodequeriesanddocumentsintodense scaletoreal-worldapplications. Asaresult,tradi-
and low-dimensional vectors (Karpukhin et al., tionallogic-basedreasoningtechniqueshavelost
2020a; Cai et al., 2022), which capture semantic popularityduetolimitedscalabilityandcoverage.
similarity instead of match of terms, thus signif- The recent breakthroughs in LLMs have
icantly outperforming sparse approaches. Rele- reignited interest in logic, bringing it back to the
vantstudiesmainlyfocusonimprovingtrainingap- forefrontofreasoningtasks. Onepromisingstrat-
proach(Quetal.,2020),distillation(Zhangetal., egytoleveragethepowerofLLMsistotranslate
2023) and pre-training (Shen et al., 2022) for re- NLstatements,suchaspremisesandconclusionsin
| trieval. |     |     |     | textualentailmenttasks,intofirst-orderlogic(FOL) |     |     |     |     |     |     |
| -------- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
Many studies adopt a transfer learning frame- formulasviain-contextlearning. Thesesymbolic
work where dense retrieval models are trained representations can be passed to Symbolic Math-
on high-resource passage retrieval datasets such ematical Theory (SMT) solvers (Olausson et al.,
as MS-MARCO (Bajaj et al., 2018) and then 2023;Xuetal.,2024)orusedtomakeveracitypre-
evaluated on queries from new tasks. However, dictionsandgenerateexplanations(WangandShu,
collecting such large-scale corpora is both time- 2023). Inthecontext,(Yangetal.,2024)presentsa
consumingandlabor-intensive. Recentworkhas NL-FOLdatasetMALLsof28Kdiverseandveri-
introducedzero-shotdenseretrievalsettings,which fiedsentence-levelpairscollectedfromGPT4,and
eliminates the need for relevance labels between atranslatorLOGICLLAMA,aLLaMA2-7B/13B
queries and documents (Gao et al., 2022). Our fine-tunedonMALLSforNL-FOLtranslation. In
workfollowsthezero-shotunsupervisedsetupfor this paper, we use LLMs as translators to imple-
| allexperiments.           |     |     |     | mentNL-FOLtranslation. |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- |
| 2.2 OptimalTransportinNLP |     |     |     | 3 Preliminaries        |     |     |     |     |     |     |
Optimaltransport(OT)hasbeenemployedinvari- 3.1 TaskFormulation
ousNLPtasks,wherealignmentexistsimplicitly
|                |             |              |           | In this | paper, | we focus | on  | the task | of  | zero-shot |
| -------------- | ----------- | ------------ | --------- | ------- | ------ | -------- | --- | -------- | --- | --------- |
| or explicitly. | The typical | applications | of OT in- |         |        |          |     |          |     |           |
documentretrieval,ofwhichthemodelcapturesthe
cludeevaluatingthesimilaritybetweensentences
similaritybetweenqueriesanddocumentswithout
anddocuments(Wangetal.,2022;Mysoreetal.,
|           |               |             |              | modeltraining.                   |     | Givenaqueryqandthedocument |     |     |           |     |
| --------- | ------------- | ----------- | ------------ | -------------------------------- | --- | -------------------------- | --- | --- | --------- | --- |
| 2021; Lee | et al., 2022) | or aligning | cross-domain |                                  |     |                            |     |     |           |     |
|           |               |             |              | setDcontainingmultipledocuments, |     |                            |     |     | thegoalof |     |
representationsacrossdifferentmodalities(Zhou
|                            |     |                    |     | retrievers                 | is  | to retrieve | document |                    | d that | satisfies |
| -------------------------- | --- | ------------------ | --- | -------------------------- | --- | ----------- | -------- | ------------------ | ------ | --------- |
| etal.,2023;Qiuetal.,2023). |     | Theevaluationmech- |     |                            |     |             |          |                    |        |           |
|                            |     |                    |     | theuser’srealsearchintent. |     |             |          | Denseretrievaluses |        |           |
anismcanbeintegratedasapenaltytermintolan-
encoderstomapqanddintoapairofdensevectors,
guagegenerationmodels(Chenetal.,2019;Zhang
|               |            |                  |        | whose | inner | product | is leveraged |     | as a similarity |     |
| ------------- | ---------- | ---------------- | ------ | ----- | ----- | ------- | ------------ | --- | --------------- | --- |
| et al., 2020; | Li et al., | 2020). Moreover, | OT ef- |       |       |         |              |     |                 |     |
function:
fectivelyhandlesimbalancedwordalignment,in-
cludingbothexplicitalignmentandnullalignment
|                    |                            |     |     |     | sim(q,d) |     | = E | q (q),E d | (d) . | (1) |
| ------------------ | -------------------------- | --- | --- | --- | -------- | --- | --- | --------- | ----- | --- |
|                    |                            |     |     |     |          |     | ⟨   |           | ⟩     |     |
| (Araseetal.,2023). | Inspiredbyunsupervisedword |     |     |     |          |     |     |           |       |     |
alignment(Araseetal.,2023;Huangetal.,2024), In this paper, we use the BGE model as query
weutilizethealignmentmatrixtomeasuredistri- encoder E and document encoder E , and the
|     |     |     |     |     | q   |     |     |     | d   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bution differences between natural language and embeddingsofCLStoken(thefirsttokenofase-
|                                                |     |     |     | quence) | denotes | dense | vectors | E   | (q) and | E (d). |
| ---------------------------------------------- | --- | --- | --- | ------- | ------- | ----- | ------- | --- | ------- | ------ |
| first-orderlogic,enablingnaturallanguagetobet- |     |     |     |         |         |       |         |     | q       | d      |
terfocusonthelogicalsemanticsinherentinfirst- Besides, we obtain word embeddings of NL and
| orderlogic.           |     |     |     | FOL sequences |     | via | BGE, | respectively. |        | Let H = |
| --------------------- | --- | --- | --- | ------------- | --- | --- | ---- | ------------- | ------ | ------- |
|                       |     |     |     | [h ,...,h     | ](h |     | d,1  | i             | m) and | =       |
|                       |     |     |     | 1             | m   | i   | R    |               |        | Z       |
|                       |     |     |     |               |     | ∈   |      | ≤ ≤           |        |         |
| 2.3 NL-FOLTranslation |     |     |     | [z ,...,z     | ](z | R   | d,1  | i n)          | be the | embed-  |
|                       |     |     |     | 1             | n   | j   |      |               |        |         |
|                       |     |     |     |               |     | ∈   | ≤    | ≤             |        |         |
NL-FOL(NaturalLanguagetoFirst-OrderLogic) dingsofNL-queriesandFOL-queries,respectively.
translationhaslongbeenachallengeinbothnatu- WeobtaintheembeddingsofNL-documentsand
|     |     |     |     | FOL-documents |     | in  | the same | way3. | We  | provide |
| --- | --- | --- | --- | ------------- | --- | --- | -------- | ----- | --- | ------- |
rallanguageprocessing(NLP)andformallogicre-
search. Traditionally,NL-FOLtranslationhasbeen
3Inthepaper,NL-queryandFOL-queryrefertoqueriesin
approachedthroughrule-basedmethods(Abzian- naturallanguageandfirst-orderlogic,respectively.Similarly,
1830

hcls
H
|     |      | NL-query |     |     |     |     |     |     |     | Connective Constraint |     |     |
| --- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- |
|     |      |          |     |     |     |     |     | …   |     |                       |     | h   |
|     | User |          |     |     |     |     |     |     | P   |                       |     |     |
OT
Logic Alignment
cls
|     |     |     |     |     | FOL-  |     |     |     |     | h   |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     | query |     |     | …   |     |     |     |     |
Z
Score
|     | BGE Retriever |     |     | GPT-4o |     |     | BGE |     |     | 1   | Score | 2 Score |
| --- | ------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | ----- | ------- |
Z
|     |     |     |     |     |     |     |     |     |     | d   | cls |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
…
Logic Alignment
|     |     |     |     | FOL-document |     |     |     |     | P   |     |     |     |
| --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
OT
|     |     | Document (NL-document) d |     |     |     |     |     |     |     |     |     | d   |
| --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Connective Constraint
…
H
Top-K documents
Figure3: ThepipelineofourproposedNS-IR.Dashedarrowsrepresenttheretrievalstage. Inthefigure,onlyone
documentdintop-K documentsisencoded,butactually,alltop-K documentsareencodedtogether.
| someexamplesofFOLinAppendixA. |                  |     |     |     |     |     | 4 Methodology |     |     |     |     |     |
| ----------------------------- | ---------------- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- |
| 3.2                           | OptimalTransport |     |     |     |     |     | 4.1 Overview  |     |     |     |     |     |
Optimaltransport(OT)seekstofindthemosteffi- The pipeline of our NS-IR is shown in Figure 3.
cientwaytotransportoneprobabilitydistributionµ ToreducethecostofNL-FOLtranslation,wefirst
| (thesource)toanotherν |     |     | (thetarget)whileminimiz- |     |     |     |         |           |     |                    |     |       |
| --------------------- | --- | --- | ------------------------ | --- | --- | --- | ------- | --------- | --- | ------------------ | --- | ----- |
|                       |     |     |                          |     |     |     | use BGE | retriever | to  | initially retrieve | the | top-K |
ingapredefinedcostfunction(Redkoetal.,2019). documentsD = d ,...,d ,...,d
|     |     |     |     |     |     |     |     |     | 1   | i   | K foragiven |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- |
|     |     |     |     |     |     |     |     |     | {   |     | }           |     |
Formally, letµandν beprobabilitymeasureson query of NL (denoted as NL-query). Then, in-
|     | X   | Y,  |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
spaces and respectively, and let function spiredbypreviouswork(Olaussonetal.,2023;Xu
| c(  | , ) represent | the cost | of  | transporting | a unit | of  |     |     |     |     |     |     |
| --- | ------------- | -------- | --- | ------------ | ------ | --- | --- | --- | --- | --- | --- | --- |
etal.,2024),weusethespecificpromptsdetailed
· ·
mass from points. The following explanation as- in Appendix B to let an LLM (GPT-4o) perform
sumesthatthesourceandtargetsentencesH and NL-FOLtranslation,soastoobtainthequeryand
| Zandtheirwordembeddingsareathand. |     |     |     |     | Acost |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
documentofFOL(denotedasFOL-queryandFOL-
| meansadissimilaritybetweenh |     |     |     | andz | (NLand |     |                        |     |     | Next,weemployBGE4 |     |     |
| --------------------------- | --- | --- | --- | ---- | ------ | --- | ---------------------- | --- | --- | ----------------- | --- | --- |
|                             |     |     |     | i    | j      |     | documentrespectively). |     |     |                   |     | to  |
FOL word embeddings) computed by a distance encodetheNL-query,FOL-query,NL-document,
| metricc | :   | d d |     |     |     |     |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
R R R+ ,suchascosinedistances. and FOL-document to obtain corresponding em-
|     |      | × →      |      |           |     |       |           |                                   |     |     |     |     |
| --- | ---- | -------- | ---- | --------- | --- | ----- | --------- | --------------------------------- | --- | --- | --- | --- |
| The | cost | matrix C | m n  | summaries | the | costs |           |                                   |     |     |     |     |
|     |      |          | R +× |           |     |       | beddings. | Finally,weintroducetwoindependent |     |     |     |     |
∈
of any word pairs, that is, C = c(h ,z ). The techniques: logicalignment(Sec.4.2)andconnec-
|     |     |     |     | i,j | i j |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
OTproblemidentifiesanalignmentmatrixPwith tiveconstraint(Sec.,4.3)torecalculatethescores
| which | the | sum of alignment |     | costs | is minimized |     |     |     |     |     |     |     |
| ----- | --- | ---------------- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- |
betweenqueriesanddocumentsandrerankcandi-
| underthecostmatrixC: |     |        |     |       |     |     | datedocuments5.    |     |     |     |     |     |
| -------------------- | --- | ------ | --- | ----- | --- | --- | ------------------ | --- | --- | --- | --- | --- |
|                      |     | L :=   | m i | n C,P | ,   | (2) |                    |     |     |     |     |     |
|                      |     | C(µ,ν) |     | ,ν)⟨  | ⟩   |     | 4.2 LogicAlignment |     |     |     |     |     |
P U( µ
∈
|       |        |           |                        |      |       |     | To incorporate |                  | overall | logic semantics |       | in FOL   |
| ----- | ------ | --------- | ---------------------- | ---- | ----- | --- | -------------- | ---------------- | ------- | --------------- | ----- | -------- |
| where | U(µ,ν) | :=        | P                      | m n  | : P 1 | =   |                |                  |         |                 |       |          |
|       |        |           |                        | R +× |       | m   | into NL        | representations, |         | we propose      | logic | align-   |
|       | 1      |           | { ∈                    |      |       |     |                |                  |         |                 |       |          |
| a,P⊤  |        | = b . P = | 0ifthei-thsourcewordis |      |       |     |                |                  |         |                 |       |          |
|       | n      | i,j       |                        |      |       |     | ment based     | on               | optimal | transport       | (OT)  | which is |
|       |        | } ̸       |                        |      |       |     |                |                  |         |                 |       |          |
alignedtothej-thtargetword,suchthatthealigned inspired by unsupervised word alignment (Arase
wordshavethesmallestdistanceinthecostmatrix
C. With this formulation, we can seek alignment 4Significantly,weuseBGEtwicefordifferentpurposes.
ThefirsttimeistoretrieveTop-Kdocuments,andthesecond
matrixP.
timeistoencodeNLandFOL.
NL-documentsandFOL-documentscorrespondtodocuments 5Inthispaper,ifnotspecificallystated,queriesanddocu-
innaturallanguageandfirst-orderlogic,respectively. mentsdenoteNL-queriesandNL-documents,respectively.
1831

Formulation QueryExample #Query #Pos. #Neg. #Irr.
Introduce Allen Ginsberg’s works , but do not mention
A-a (A) 136 136 136
’Howl’ .
(a)
WhatthemesareexpressedinAllenGinsberg’sworks other
(A-a) B (A) 123 123 123
∪ than’Howl’ (a) andEdgarAllanPoe’sworks (B) ? 3000
WhatthemesdoAllenGinsberg’sworks otherthan’Howl’
(A) (a)
(A-a) ∪ (B-b) andEdgarAllanPoe’sworks (B) otherthan’TheRaven’ (b) ex- 107 107 321
press?
Table1: Examplesofnegative-constraintqueriesinourconstructeddatasetNegConstraint. #Querydenotesthe
numberofeachtypeofquery,#Pos.,#Neg.,and#Irr. denotethenumberofpositivedocuments,andirrelevant
documents,respectively.
et al., 2023; Huang et al., 2024). This approach 4.3 ConnectiveConstraint
measuresthedistributiondifferencesbetweenNL
TopreciselyreflecttheroleofpartialwordsinFOL
andFOL,andintegrateswordfeaturesofNLand
onlogicalconsistency,wealsoproposeconnective
FOLwithcontextrepresentationviathealignment
constraintthatenablesdifferentwordsinFOL(es-
matrix.
peciallyconnectives)torenderdifferentattentions
Specifically, for a given NL-query and FOL-
towordsinNL,thusgeneratingbetterembeddings
query,wefirstuseBGEtoobtaintheircorrespond-
with logical semantics. Given a FOL-query se-
ingwordembeddingsHandZ6,respectively. Then,
quencet = t ,...,t ,...,t ,aswellasNL-query
1 i n
wecomputeapairwisesimilaritybetweenHandZ { }
wordembeddingsHandFOL-querywordembed-
usingcosinedistanceC :
i,j dings Z, we integrate the embeddings of logical
connectivesintotheattentions. Thatis,whencal-
hTz
C = 1 i j . (3) culating attention weights (Eq. 8) of FOL to NL
i,j
− h z
∥ i ∥∥ j ∥ andupdatedembeddings(Eq.7),ittakesthealign-
mentbetweenNLandFOLandtheembeddingsof
TheOTcanbeformulatedas:
logicalconnectivesintoaccount:
P∗ = argmin C
i,j
P
i,j
, (4)
m
P ∈ U(H,Z) X i,j h = α (h +σ z ), (7)
j ji i ji j
where alignment matrix P i,j R m × n indicates a X i=1
∈
likelihood of aligning h with z . The optimiza-
i j
tioninEq.4canbesolvedbylinearprogramming eα′ji z j (h i +σ ji z j )T
(BourgeoisandLassalle,1971).
α ji =
m eα′ji
, α j′i =
√d k
, (8)
i=1
Finally, we integrate H, Z, P and hcls R d
(hcls R d denote the embeding of special ∈ token P 1, t j = ,t j C,P ij = 0
∈ ̸ ¬ ∈
CLS7.)toobtainupdatedembeddinghcls
∈
R
d:
σ
ji
=

1, t
j
= ,P
ij
= 0 , (9)
− ¬

hcls = H T P Z h cls. (5) 0, otherwise
· · ·
wherePd enotesthealignmentmatrixandlogical
This approach synthesizes word distributions of
connectivesetC = , , , , , .
FOLandNL,aswellascontextfeatures,viaalign- {¬ → ↔ ∧ ∨ ⊕}
Ourexplanationsforaboveequationsareasfol-
mentmatrixP astheintermediary.
i,j
lows. Forthewordsthatdonotexhibitsignificant
We use the same method to obtain the embed-
alignmentbetweenFOLandNL(whereP = 0),
dingdcls R d ofadocumentdinthedocument i,j
∈ weincorporatelogicalconnectiveembeddingsto
set D, and then calculate the similarity score be-
further enhance logical semantics. Additionally,
tweenhcls anddcls as
for negative connective (t = ), the subtraction
j
¬
score = sim hcls dcls . (6) impliesthenegative-constraintsemantics.
1
· Finally,weperformmeanpoolingonh toob-
j
6HandZdenotecorrespon (cid:16) dingwordem (cid:17) beddingsofNL- tainaquery’sembeddingas
documentsandFOL-documents,respectively.
7Infact,thefirsttokenofqueriesisdenotedasCLS,i.e.,
hcls =h1. h = mean-pooling(h j ), h R d. (10)
∈
1832

Similarly,weobtaintheembeddingofadocument benchmark (Thakur et al., 2021), including Sci-
dfromthedocumentcollectionD,andthencom-
Fact(fact-checking),ArguAna(argumentretrieval),
putethesimilarityscorebetweenhandd as TREC-COVID(bio-medicalIR),FiQA(financial
|     |       |         |     |     |      | QA), DBPedia |      | (entity  | retrieval), |     | and NFCorpus |          |
| --- | ----- | ------- | --- | --- | ---- | ------------ | ---- | -------- | ----------- | --- | ------------ | -------- |
|     | score | = sim(h | d). |     | (11) |              |      |          |             |     |              |          |
|     |       | 2       |     |     |      | (medical     | IR). | For this | task,       | we  | report       | all com- |
·
paredmethods’scoresoftherepresentativemetric
Fordocumentd,thefinalrecommendedscoreis
|     |     |     |     |     |     | nDCG@10. |        | For web | search, | we  | adopt | widely-  |
| --- | --- | --- | --- | --- | --- | -------- | ------ | ------- | ------- | --- | ----- | -------- |
|     |     |     |     |     |     | used web | search | dataset | TREC    |     | Deep  | Learning |
score = score 1 +score 2 . (12) 2019 (DL’19) (Craswell et al., 2020) and Deep
|          |        |           |           |           |     | Learning | 2020     | (DL’20) | (Craswell |     | et al.,     | 2021) |
| -------- | ------ | --------- | --------- | --------- | --- | -------- | -------- | ------- | --------- | --- | ----------- | ----- |
| Then, we | rerank | the top-K | candidate | documents |     |          |          |         |           |     |             |       |
|          |        |           |           |           |     | based on | MS-MARCO |         | (Bajaj    | et  | al., 2018). | For   |
basedontherecommendationscores.
|     |     |     |     |     |     | these two | datasets | and | our | NegConstraint, |     | we re- |
| --- | --- | --- | --- | --- | --- | --------- | -------- | --- | --- | -------------- | --- | ------ |
portthemethods’scoresofMAP(MeanAverage
5 NegConstraint
Precision)andnDCG@10.
ToevaluateourNS-IR’sperformancespecifically
|     |     |     |     |     |     | 6.2 Baselines |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
tonegative-constraintqueries,wehaveconstructed
ahuman-annotateddatasetNegConstraint. Table1 We first compare our NS-IR with some fully-
liststhreeformulationsthatrepresentthreetypesof supervised retrieval methods that are fine-tuned
negative-constraintsquery. Lettheletters(A,a,B, withextensivequery-documentrelevancedata(de-
andb)denoteentitysetswherethelowercaselet- notedasw/relevancejudgment),includingDPR
tersrepresentthesubsetsofthesetsdenotedbythe (Karpukhin et al., 2020b), ANCE (Xiong et al.,
uppercaseletters. Operator‘-’indicatesanegative- 2020), and the fine-tuned Contriever (denoted as
|                      |     |      |                    |     |     | ContrieverFT |     | (Izacardetal.,2021)). |     |     | Wealsocon- |     |
| -------------------- | --- | ---- | ------------------ | --- | --- | ------------ | --- | --------------------- | --- | --- | ---------- | --- |
| constraintcondition, |     | and‘ | ’denotesunionoper- |     |     |              |     |                       |     |     |            |     |
∪
ation. Thereare136queriescorrespondingtothe siderseveralzero-shotretrievalmodelsnotinvolv-
firstformulation,123queriescorrespondingtothe ingquery-documentrelevancelabels(denotedas
second formulation, and 107 queries correspond- w/o relevance judgment), including sparse re-
ingtothethirdformulationinNegConstraint. Each triever BM25 (Robertson et al., 2009), dense re-
query is paired with one positive document and triever BGE (Xiao et al., 2024), and Contriever
oneorthreenegativedocumentsasdistractors. For (Izacardetal.,2021). Forthistypeofbaselines,we
|     |     |     |     |     |     | further | consider | the | LLM-based | retrieval |     | models |
| --- | --- | --- | --- | --- | --- | ------- | -------- | --- | --------- | --------- | --- | ------ |
example,foraqueryofformulationA–a,suchas
“IntroduceAllenGinsberg’sworks,butdonotmen- whichrewritequerieswithLLMs,includingHyDE
tion‘Howl”’,thereisapositivedocumentthatintro- (Gaoetal.,2023)andInteR(Fengetal.,2024).
ducesthecontentaboutAllenGinsberg’sworksbut
|                       |     |     |                         |     |     | 6.3 ImplementationDetails |     |     |     |     |     |     |
| --------------------- | --- | --- | ----------------------- | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- |
| doesnotmention‘Howl’. |     |     | Similarly,anegativedoc- |     |     |                           |     |     |     |     |     |     |
umentintroducesAllenGinsberg’sandmentions Weemploy"bge-large-en-v1.5"asembeddingmod-
‘Howl’. ThedocumentsstemfromtheWikipedia els. Tomakeafaircomparison,wealsoreproduce
|     |     |     |     |     |     | results on | HyDE | and | InteR | where | "bge-large-en- |     |
| --- | --- | --- | --- | --- | --- | ---------- | ---- | --- | ----- | ----- | -------------- | --- |
dump(Karpukhinetal.,2020a),andthequeriesare
generatedbyGPT-4obasedoncorrespondingneg- v1.5" is used as retriever models. We run all ex-
ative and positive documents where the prompts periments on one Nvidia A800 80GB GPU. For
|                     |     |                  |              |          |       | NL-FOL            | translation, |               | we use | OpenAI | API        | with a |
| ------------------- | --- | ---------------- | ------------ | -------- | ----- | ----------------- | ------------ | ------------- | ------ | ------ | ---------- | ------ |
| are presented       | in  | Appendix         | B. Besides,  | NegCon-  |       |                   |              |               |        |        |            |        |
| straint contains    |     | 3,000 irrelevant | documents8   |          | that  | temperatureof0.5. |              |               |        |        |            |        |
| are irrelevant      | to  | all queries.     | More         | details  | about |                   |              |               |        |        |            |        |
|                     |     |                  |              |          |       | 6.4 MainResults   |              |               |        |        |            |        |
| our data collection |     | and              | snippets are | provided | in    |                   |              |               |        |        |            |        |
|                     |     |                  |              |          |       | In the following  |              | presentation, |        | the    | techniques | of     |
AppendixC.
logicalignmentandconnectiveconstraintwepro-
| 6 Experiments |     |     |     |     |     | posed for     | NS-IR | are    | abbreviated |              | as LA | and CC, |
| ------------- | --- | --- | --- | --- | --- | ------------- | ----- | ------ | ----------- | ------------ | ----- | ------- |
|               |     |     |     |     |     | respectively. |       | In our | comparison  | experiments, |       | we      |
6.1 DatasetsandMetrics
|     |     |     |     |     |     | adopt two | variants |     | of NS-IR | which | use | Logi- |
| --- | --- | --- | --- | --- | --- | --------- | -------- | --- | -------- | ----- | --- | ----- |
For low-resource retrieval, we use six diverse cLLaMA(Yangetal.,2024)andGPT-4otogener-
| low-resource | retrieval |     | datasets from | the | BEIR |     |     |     |     |     |     |     |
| ------------ | --------- | --- | ------------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
ateFOL,respectively.
|     |     |     |     |     |     | Table | 2 shows | that, | our | NS-IR | (GPT-4o) | out- |
| --- | --- | --- | --- | --- | --- | ----- | ------- | ----- | --- | ----- | -------- | ---- |
8Negativedocumentsareessentiallyirrelevantdocuments,
buteasiertomisleadretrievermodels. performs all baselines significantly on the tasks
1833

Methods SciFact ArguAna TREC-COVID FiQA DBPedia NFCorpus DL’19 DL’20
| w/relevancejudgment |     |      |      |     | nDCG@10 |           |     | MAP       | nDCG@10 | MAP  | nDCG@10 |
| ------------------- | --- | ---- | ---- | --- | ------- | --------- | --- | --------- | ------- | ---- | ------- |
| DPR                 |     | 31.8 | 17.5 |     | 33.2    | 29.5 26.3 |     | 18.9 36.5 | 62.2    | 41.8 | 65.3    |
| ANCE                |     | 50.7 | 41.5 |     | 65.4    | 30.0 28.1 |     | 23.7 37.1 | 64.5    | 40.8 | 64.6    |
ContrieverFT 67.7 44.6 59.6 32.9 41.3 32.8 41.7 62.1 43.6 63.2
| w/orelevancejudgment |     |      |      |     | nDCG@10 |           |     | MAP       | nDCG@10 | MAP  | nDCG@10 |
| -------------------- | --- | ---- | ---- | --- | ------- | --------- | --- | --------- | ------- | ---- | ------- |
| BM25                 |     | 67.1 | 43.2 |     | 55.5    | 25.1 26.1 |     | 31.4 31.2 | 55.4    | 30.6 | 50.1    |
♡
| Contriever |     | 55.0 | 44.5 |     | 12.5 | 12.4 29.2 |     | 26.0 22.8 | 37.5 | 24.3 | 42.5 |
| ---------- | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡
| HyDE |     | 71.9 | 49.6 |     | 78.4 | 31.3 38.7 |     | 37.3 48.7 | 67.3 | 49.8 | 66.8 |
| ---- | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡
| InterR |     | 72.1 | 50.9 |     | 79.2 | 33.5 42.1 |     | 39.5 50.4 | 69.7 | 47.8 | 67.5 |
| ------ | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡
| BGE |     | 71.3 | 48.4 |     | 75.3 | 30.6 38.9 |     | 35.4 46.9 | 64.4 | 45.7 | 63.4 |
| --- | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡∗
| BGEw/LA |     | 72.6 | 53.2 |     | 78.3 | 33.7 42.8 |     | 38.1 48.9 | 67.5 | 47.5 | 68.9 |
| ------- | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡∗
| BGEw/CC |     | 73.3 | 52.3 |     | 77.6 | 35.6 43.2 |     | 37.7 49.1 | 66.8 | 48.5 | 66.6 |
| ------- | --- | ---- | ---- | --- | ---- | --------- | --- | --------- | ---- | ---- | ---- |
♡∗
NS-IR(LogicLLaMA) 73.7 51.1 78.8 35.5 42.6 38.8 49.8 67.9 48.9 68.1
NS-IR(GPT-4o) 75.8 55.1 81.8 38.4 46.1 40.7 51.4 68.4 50.8 70.5
Table 2: Performance of compared methods on the benchmarks of low-resource retrieval and web search.
♡
indicatesthereportedresultswerereproducedbyususingthebaselines’sourcecodes. WeemployBGEasthe
embeddingmodelinHyDEandInteRforfaircomparison. denotestheablatedvariantsofNS-IRwhichcanbe
∗
regardedasBGEw/LA&CC.
|         |     |     | A-a     |      | (A-a) | B       | (A-a) | (B-b)   |      | Total   |      |
| ------- | --- | --- | ------- | ---- | ----- | ------- | ----- | ------- | ---- | ------- | ---- |
| Methods |     |     |         |      |       | ∪       |       | ∪       |      |         |      |
|         |     | MAP | nDCG@10 |      | MAP   | nDCG@10 | MAP   | nDCG@10 | MAP  | nDCG@10 |      |
| BM25    |     |     | 32.1    | 34.6 | 31.2  | 34.7    | 29.3  | 31.5    | 31.4 |         | 33.7 |
♡
| Contriever |     |     | 34.8 | 36.6 | 32.1 | 33.3 | 30.9 | 32.7 | 31.8 |     | 35.7 |
| ---------- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- |
♡
| HyDE |     |     | 50.7 | 55.3 | 48.6 | 51.5 | 45.7 | 50.6 | 47.8 |     | 53.1 |
| ---- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- |
♡
| InterR |     |     | 52.6 | 55.8 | 50.3 | 48.7 | 51.5 | 49.3 | 52.3 |     | 54.5 |
| ------ | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- |
| BGE    | ♡   |     | 37.9 | 40.5 | 34.8 | 36.8 | 33.7 | 34.8 | 36.3 |     | 40.8 |
∗
| BGEw/LA | ♡   |     | 42.1 | 45.6 | 41.2 | 44.6 | 39.9 | 42.5 | 40.8 |     | 47.6 |
| ------- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- |
∗
| BGEw/CC | ♡   |     | 48.9 | 50.7 | 46.6 | 48.5 | 43.9 | 48.2 | 47.8 |     | 46.9 |
| ------- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- |
∗
♡
| NS-IR(LogicLLaMA) |     |           | 53.2                                                     | 54.6    | 51.6    | 50.2           | 49.6 | 54.1 | 50.7 |     | 55.2 |
| ----------------- | --- | --------- | -------------------------------------------------------- | ------- | ------- | -------------- | ---- | ---- | ---- | --- | ---- |
| NS-IR(GPT-4o)     |     |           | 54.7                                                     | 57.9    | 53.3    | 54.2           | 51.7 | 53.7 | 53.3 |     | 56.5 |
|                   |     | Table3:   | PerformancecomparisonsofdifferentmethodsonNegConstraint. |         |         |                |      |      |      |     |      |
| of low-resource   |     | retrieval | and web                                                  | search, | includ- | moreeffective. |      |      |      |     |      |
ingtheSOTAmodelwithoutrelevancejudgment The results in Tables 2 and 3 related to NS-
| InteR. Specifically, |     | NS-IR | (GPT-4o) | obtains | an  |              |     |                  |     |             |         |
| -------------------- | --- | ----- | -------- | ------- | --- | ------------ | --- | ---------------- | --- | ----------- | ------- |
|                      |     |       |          |         |     | IR’s ablated |     | variants (marked |     | by ∗ ) also | justify |
averageperformanceimprovementofover10%rel- the effectiveness of employing either LA or CC.
ativetothevanillaBGE.TheinferiorityofNS-IR Inparticular,adoptingCCimprovesNS-IR’sper-
(LogicLLaMA) compared to NS-IR (GPT-4o) is formance more obviously than adopting LA on
attributedtoLogicLLaMA’sweaknessonNL-FOL NegConstraint,suggestingthatCCismoreeffec-
| translation.                               |     |     |     |     |     | tivethanLAinthescenariosofnegative-constraint |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- |
| Fornegative-constraintqueries,wecompareNS- |     |     |     |     |     | queries.                                      |     |     |     |     |     |
IRanditsablatedvariantswiththebaselineswith-
|                       |     |                      |     |       |          | 6.5 EffectsofDifferentDenseRetrievers |     |               |     |           |           |
| --------------------- | --- | -------------------- | --- | ----- | -------- | ------------------------------------- | --- | ------------- | --- | --------- | --------- |
| outrelevancejudgment. |     | Table3reportsthecom- |     |       |          |                                       |     |               |     |           |           |
|                       |     |                      |     |       |          | To verify                             | the | effectiveness | of  | different | dense re- |
| pared methods’        |     | performance          | on  | three | types of |                                       |     |               |     |           |           |
negative-constraint queries and whole queries in trievers,wereportthewebsearchperformanceof
NegConstraint. Theresultsrevealourmethod’ssu- HyDE,InteR,andNS-IR(GPT-4o)withdifferent
|     |     |     |     |     |     | dense | retrievers | (bge-small, | bge-base, |     | and Con- |
| --- | --- | --- | --- | --- | --- | ----- | ---------- | ----------- | --------- | --- | -------- |
periorityoverthebaselinesonnegative-constraint
queries, which is achieved through synthesizing triver)9 inTable4. Theresultsindicatethatmore
semanticsimilarityandlogicalconsistencyforhan- powerful retriever models can facilitate accurate
dling complex logical queries. Although HyDE IR. NS-IR is consistently superior to HyDE and
|            |             |                     |     |            |     | InteRwithallretrievers. |     |     | Theseresultsalsoindicate |     |     |
| ---------- | ----------- | ------------------- | --- | ---------- | --- | ----------------------- | --- | --- | ------------------------ | --- | --- |
| and InterR | can         | partially eliminate |     | the impact | of  |                         |     |     |                          |     |     |
| negative   | constraints | via hypothetical    |     | documents  |     |                         |     |     |                          |     |     |
9Wereplacedenseretrieversintheprocessofretrievaland
generatedbyLLMs,ourproposedLAandCCare encodingasintroducedinSec.4.1.
1834

| Query | Positive document |     | Negative document |     |     |     |     |     |     |     |
| ----- | ----------------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
(a) Vanilla BGE (b) BGE w/ logic alignment (c) BGE w/ connective constraint
Figure4: AnexampleofqueryembeddingvisualizationfromTREC-COVID(betterviewedincolor): Whatarethe
observedmutationsintheSARS-CoV-2genomeandhowoftendothemutationsoccur?
Query Positive document Negative document Irrelevant document
(a) Vanilla BGE (b) BGE w/ logic alignment (c) BGE w/ connective constraint
Figure5: AnexampleofqueryembeddingvisualizationfromNegConstraint(betterviewedincolor): Whatarethe
similaritiesbetweenGinsberg’sworks(excluding’Howl’)andPoe’sworks(excluding’TheRaven’)?
|     |     | DL’19 |     | DL’20 | LAandCC.InFigures4and5,weplottheembed- |     |     |     |     |     |
| --- | --- | ----- | --- | ----- | -------------------------------------- | --- | --- | --- | --- | --- |
Methods
dingsgeneratedbyvanillaBGE,BGEw/LAand
|     | MAP | nDCG@10 | MAP | nDCG@10 |     |     |     |     |     |     |
| --- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- |
BGEw/CCintheembeddingspaceusingt-SNE,
| bge-small | 40.5 | 60.1 | 40.4 | 61.2 |               |                             |     |     |     |     |
| --------- | ---- | ---- | ---- | ---- | ------------- | --------------------------- | --- | --- | --- | --- |
|           |      |      |      |      | respectively. | InFigure4ofTREC-COVID,wecan |     |     |     |     |
| +HyDE     | 42.7 | 61.4 | 41.8 | 62.7 |               |                             |     |     |     |     |
+InteR 43.6 63.4 42.8 63.2 seethatthequeryembeddingsgeneratedbyBGE
| +NS-IR | 43.8 | 65.4 | 44.4 | 64.9 |     |     |     |     |     |     |
| ------ | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- |
w/LAandBGEw/CCareclosertothatofpositive
bge-base 41.9 62.5 40.9 63.1 documentsthanthequeryembeddingsgenerated
| +HyDE | 42.4 | 64.4 | 42.6 | 64.5 |     |     |     |     |     |     |
| ----- | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- |
byvanillaBGE.InFigure5ofNegConstraint,the
| +InteR | 45.9 | 65.3 | 45.5 | 66.7 |     |     |     |     |     |     |
| ------ | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- |
+NS-IR 46.6 66.4 47.9 68.8 queryembeddingsgeneratedbyBGEw/LAand
Contriever 35.7 57.7 37.8 56.9 BGEw/CCareclosertothatofpositivedocuments
+HyDE 37.5 59.3 39.9 58.9 andfartherawayfromthatofnegativedocuments,
| +InteR | 38.6 | 58.9 | 39.8 | 61.7 |              |        |         |             |     |              |
| ------ | ---- | ---- | ---- | ---- | ------------ | ------ | ------- | ----------- | --- | ------------ |
|        |      |      |      |      | compared     | to the | query   | embeddings  |     | generated by |
| +NS-IR | 40.6 | 61.4 | 41.4 | 62.8 |              |        |         |             |     |              |
|        |      |      |      |      | vanilla BGE. | These  | results | demonstrate |     | that LA      |
ancCCaremoreeffectiveonidentifyingpositive
Table4: Websearchperformanceofadoptingdifferent
| denseretrievermodelsinHyDE,InteRandourNS-IR. |     |     |     |     | documents. |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
6.7 VisualizationontheAttentionof
| that the effectiveness |     | of LA | and CC | on NS-IR’s |     |     |     |     |     |     |
| ---------------------- | --- | ----- | ------ | ---------- | --- | --- | --- | --- | --- | --- |
ConnectiveConstraint
performancegainsaremodel-agnostic.
|     |     |     |     |     | As introduced | in  | Sec. | 4.2, CC | enables | the words |
| --- | --- | --- | --- | --- | ------------- | --- | ---- | ------- | ------- | --------- |
6.6 VisualizationontheEffectsofLogic
|     |     |     |     |     | in FOL (especially |     | connectives) |     | to  | assign differ- |
| --- | --- | --- | --- | --- | ------------------ | --- | ------------ | --- | --- | -------------- |
AlignmentandConnectiveConstraint
entattentionstodifferentwordsinNL.Toverify
WerandomlypicktwoqueriesfromTREC-COVID the hypothesis, we examine the attention scores
and our NegConstraint to visualize the effects of oflogicalnegation inFOLto thewordsinNL.
¬
1835

Query: What are the  similarities between Ginsberg's works  stabilized,indicatingthatincreasedK didnotsig-
(excluding ‘Howl’) and Poe's works (excluding ‘The Raven’)?
|     |     |     |     |     | nificantly | enhance | the | results. | This | phenomenon |     |
| --- | --- | --- | --- | --- | ---------- | ------- | --- | -------- | ---- | ---------- | --- |

|           |            |                 |                    |     | can be attributed |     | to the | fact | that the | first 100 | re- |
| --------- | ---------- | --------------- | ------------------ | --- | ----------------- | --- | ------ | ---- | -------- | --------- | --- |
| Positive  | Document:  | Ginsberg  took  | part  in  decades  | of  |                   |     |        |      |          |           |     |
non-violent political protest against everything from the Vietnam  trieved documents have already covered a signif-
War to the War on Drugs. His poem ""September on Jessore
Road"", calling attention to the plight of Bangladeshi refugees,  icant amount of positive documents for a query.
exemplifies what the literary critic Helen Vendler described as  Additionally,thelargernumberofretrievaldocu-
Ginsberg's tireless persistence…, though he died before it could
mentswillextremelyincreaseexpensesforgener-
be produced. by gambling, and the cost of secondary education
for Poe. He attended the University of Virginia but left after a year  ating FOL. Therefore, we select 100 as Top K in
| due to lack of money. Poe quarreled with Allan over the funds for  |     |     |     |     | thispaper. |     |     |     |     |     |     |
| ------------------------------------------------------------------ | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
his education and enlisted in the Army in 1827 under an assumed
name…Allan reached a temporary rapprochement. However, Poe
later failed as an officer cadet at West Point, declaring…
|     |     |     |     |     | 8 Conclusion |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
Figure 6: Attention scores of logical connective in In this paper, we propose a novel IR method NS-
¬
| FOL | to the words | in NL. A deeper | color indicates | a   |     |     |     |     |     |     |     |
| --- | ------------ | --------------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
IR,whichintegratesthestrengthsofNLandFOL
biggerscore(betterviewedincolor).
|     |     |     |     |     | and synthesizes |     | semantic  | similarity |     | and     | logical |
| --- | --- | --- | --- | --- | --------------- | --- | --------- | ---------- | --- | ------- | ------- |
|     |     |     |     |     | consistency.    | We  | specially | propose    |     | two key | tech-   |
As shown in Figure 6, the deeper colors indicate niques: logicalignmentandconnectiveconstraint,
the larger attention scores. We suppose that this torerankthecandidatedocuments. Wealsorelease
operationemphasizesimportantentitytokens(such
anegative-constraintquerydatasetNegConstraint
as‘Ginsberg’and‘Poe’)andignoresentitytokens to evaluate our method. Extensive experiments
in negative-constraint conditions (such as ‘Howl’ onpublicIRbenchmarksandNegConstraintshow
and ‘Raven’). That is, logical connective im- that,NS-IRsignificantlyoutperformstheexisting
¬
| pliesnegative-constraintsemantics. |     |     | Itrevealsthat |     |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
IRapproachesforgeneralandnegative-constraint
ourmethodtendstoretrievethedocumentswith- queriesunderzero-shotsettings,pavingthewayfor
out negative-constraint conditions mentioned in futurestudyoncomplexlogicalqueries. Therefore,
queries.
wewillfocusonmorecomplexlogicalqueriesgen-
eratedbysetoperations(suchasunion,intersection,
7 EffectofParameterK
difference,andcomplement)inthefuture.
|     |     |     |     |     | 9 Acknowledgments |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
 6 F L ) D F W
 $ U J X $ Q D The authors disclosed receipt of the following fi-
|     |    |     |     |     | nancial support           |     | for the | research,           | authorship, |     | and |
| --- | ---- | --- | --- | --- | ------------------------- | --- | ------- | ------------------- | ----------- | --- | --- |
|     |      |     |     |     | publicationofthisarticle: |     |         | Thisresearchwassup- |             |     |     |
   # * & ' Q    portedbytheAILaboratoryofUnitedAutomotive
|     |      |     |     |     | ElectronicSystems(UAES)Co.             |     |     |     | (Grantno. |     | 2025- |
| --- | ---- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --------- | --- | ----- |
|     |    |     |     |     | 3944)andtheChineseNSFMajorResearchPlan |     |     |     |           |     |       |
(No.92270121).
  
Limitations
|     |    |       |         |     |     |     |     |     |     |     |     |
| --- | --- | --------- | ------------- | ------ | --- | --- | --- | --- | --- | --- | --- |
 7 R S  .
Weacknowledgethatourmethodhasseverallimi-
Figure7: TheperformanceofNS-IRondifferentTopK tations. First,callingOpenAIAPItoperformNL-
ofSciFactandArguAna. FOL translation will inevitably incur additional
|     |     |     |     |     | expensestomaintainhighretrievalrelevance. |     |     |     |     |     | Sec- |
| --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | ---- |
We perform an additional study to investigate ond,toreducetheexpensesofNL-FOLtranslation,
theimpactofthenumberofretrieveddocuments we perform NL-FOL translation on the initially
(i.e.,TopK)ontheperformanceofNS-IR.Figure7 retrievedand limited documents, thus slightlyre-
illustratesnDCG@10underdifferentK onSciFact ducing recall. Third, we use the same prompts
and ArguAna. Our observations revealed consis- forNL-FOLtranslationonallbenchmarks,which
tentpatternsinbothdatasets: asTopK increased, mayhinderfurtherimprovement. Therefore,these
performance showed a gradual improvement un- limitationsarecausedbyNL-FOLtranslation. Al-
tilK reached100. Subsequently,theperformance thoughNL-FOLtranslationisnotthemainfocus
1836

ofthispaper,wearguethatthelimitationswillbe pages10028–10039,Bangkok,Thailand.Association
forComputationalLinguistics.
improvedwithfurtherstudyintheeraofLLMs.
JiazhanFeng,ChongyangTao,XiuboGeng,TaoShen,
CanXu,GuodongLong,DongyanZhao,andDaxin
References
|     |     |     |     | Jiang. 2024. | Synergistic | interplay | between | search |
| --- | --- | --- | --- | ------------ | ----------- | --------- | ------- | ------ |
LashaAbzianidze.2017. Langpro: Naturallanguage andlargelanguagemodelsforinformationretrieval.
|                |                                |     |     | In Proceedings | of the | 62nd Annual | Meeting | of the |
| -------------- | ------------------------------ | --- | --- | -------------- | ------ | ----------- | ------- | ------ |
| theoremprover. | arXivpreprintarXiv:1708.09417. |     |     |                |        |             |         |        |
AssociationforComputationalLinguistics(Volume1:
Yuki Arase, Han Bao, and Sho Yokoi. 2023. Unbal- LongPapers),pages9571–9583,Bangkok,Thailand.
ancedoptimaltransportforunbalancedwordalign- AssociationforComputationalLinguistics.
ment. InProceedingsofthe61stAnnualMeetingof
theAssociationforComputationalLinguistics(Vol- LuyuGao,XueguangMa,JimmyLin,andJamieCallan.
ume 1: Long Papers), pages 3966–3986, Toronto, 2022. Precisezero-shotdenseretrievalwithoutrele-
Canada.AssociationforComputationalLinguistics.
|     |     |     |     | vancelabels. | arXivpreprintarXiv:2212.10496. |     |     |     |
| --- | --- | --- | --- | ------------ | ------------------------------ | --- | --- | --- |
PBajaj,DCampos,NCraswell,LDeng,JGao,XLiu,
LuyuGao,XueguangMa,JimmyLin,andJamieCallan.
RMajumder,AMcNamara,BMitra,TNguyen,etal. 2023. Precisezero-shotdenseretrievalwithoutrel-
2018. Ahumangeneratedmachinereadingcompre- evance labels. In Proceedings of the 61st Annual
| hensiondataset.  | arXivpreprintarXiv:1611.09268.    |     |     |                   |                             |     |               |      |
| ---------------- | --------------------------------- | --- | --- | ----------------- | --------------------------- | --- | ------------- | ---- |
|                  |                                   |     |     | Meeting           | of the Association          | for | Computational | Lin- |
|                  |                                   |     |     | guistics(Volume1: | LongPapers),pages1762–1777, |     |               |      |
| JonBarwise.1977. | Anintroductiontofirst-orderlogic. |     |     |                   |                             |     |               |      |
Toronto,Canada.AssociationforComputationalLin-
| In Studies | in Logic and | the Foundations | of Mathe- |     |     |     |     |     |
| ---------- | ------------ | --------------- | --------- | --- | --- | --- | --- | --- |
guistics.
matics,volume90,pages5–46.Elsevier.
FrancoisBourgeoisandJean-ClaudeLassalle.1971. An Chenyang Huang, Abbas Ghaddar, Ivan Kobyzev,
extension of the munkres algorithm for the assign- MehdiRezagholizadeh,OsmarZaiane,andBoxing
|                                   |     |     |            | Chen.2024.   | OTTAWA:OptimalTransporTadaptive |     |              |        |
| --------------------------------- | --- | --- | ---------- | ------------ | ------------------------------- | --- | ------------ | ------ |
| mentproblemtorectangularmatrices. |     |     | Communica- |              |                                 |     |              |        |
|                                   |     |     |            | word aligner | for hallucination               |     | and omission | trans- |
tionsoftheACM,14(12):802–804.
|     |     |     |     | lation errors | detection. | In Findings | of  | the Asso- |
| --- | --- | --- | --- | ------------- | ---------- | ----------- | --- | --------- |
ZeFeng Cai, Chongyang Tao, Tao Shen, Can Xu, Xi- ciation for Computational Linguistics: ACL 2024,
uboGeng,XinAlexLin,LiangHe,andDaxinJiang. pages6322–6334,Bangkok,Thailand.Association
2022. Hyper: Multitask hyper-prompted training forComputationalLinguistics.
| enableslarge-scaleretrievalgeneralization. |     |     | InThe |     |     |     |     |     |
| ------------------------------------------ | --- | --- | ----- | --- | --- | --- | --- | --- |
EleventhInternationalConferenceonLearningRep- OzHuly,IdanPogrebinsky,DavidCarmel,OrenKur-
| resentations. |     |     |     | land,andYoelleMaarek.2024. |     |     | Oldirmethodsmeet |     |
| ------------- | --- | --- | --- | -------------------------- | --- | --- | ---------------- | --- |
rag. InProceedingsofthe47thInternationalACM
DanqiChen,AdamFisch,JasonWeston,andAntoine SIGIRConferenceonResearchandDevelopmentin
Bordes.2017. ReadingWikipediatoansweropen- InformationRetrieval,pages2559–2563.
| domainquestions. | InProceedingsofthe55thAnnual |     |     |     |     |     |     |     |
| ---------------- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- |
Meeting of the Association for Computational Lin- GautierIzacard,MathildeCaron,LucasHosseini,Se-
| guistics(Volume1: | LongPapers),pages1870–1879, |     |     |         |               |             |        |         |
| ----------------- | --------------------------- | --- | --- | ------- | ------------- | ----------- | ------ | ------- |
|                   |                             |     |     | bastian | Riedel, Piotr | Bojanowski, | Armand | Joulin, |
Vancouver,Canada.AssociationforComputational andEdouardGrave.2021. Unsuperviseddensein-
Linguistics. formationretrievalwithcontrastivelearning. arXiv
preprintarXiv:2112.09118.
LiqunChen,YizheZhang,RuiyiZhang,ChenyangTao,
| Zhe Gan, | Haichao Zhang, | Bai Li, Dinghan | Shen, |     |     |     |     |     |
| -------- | -------------- | --------------- | ----- | --- | --- | --- | --- | --- |
VladimirKarpukhin,BarlasOguz,SewonMin,Patrick
| Changyou | Chen, and Lawrence | Carin. | 2019. Im- |     |     |     |     |     |
| -------- | ------------------ | ------ | --------- | --- | --- | --- | --- | --- |
Lewis,LedellWu,SergeyEdunov,DanqiChen,and
provingsequence-to-sequencelearningviaoptimal
|                |                                |       |             | Wen-tau     | Yih. 2020a. | Dense passage | retrieval      | for |
| -------------- | ------------------------------ | ----- | ----------- | ----------- | ----------- | ------------- | -------------- | --- |
| transport.     | arXivpreprintarXiv:1901.06283. |       |             |             |             |               |                |     |
|                |                                |       |             | open-domain | question    | answering.    | In Proceedings |     |
|                |                                |       |             | of the 2020 | Conference  | on Empirical  | Methods        | in  |
| Nick Craswell, | Bhaskar Mitra,                 | Emine | Yilmaz, and |             |             |               |                |     |
NaturalLanguageProcessing(EMNLP),pages6769–
| Daniel Campos.     | 2021.                      | Overview of | the trec 2020 |               |             |     |               |      |
| ------------------ | -------------------------- | ----------- | ------------- | ------------- | ----------- | --- | ------------- | ---- |
|                    |                            |             |               | 6781, Online. | Association | for | Computational | Lin- |
| deeplearningtrack. | Preprint,arXiv:2102.07662. |             |               |               |             |     |               |      |
guistics.
NickCraswell,BhaskarMitra,EmineYilmaz,Daniel
VladimirKarpukhin,BarlasOg˘uz,SewonMin,Patrick
| Campos, | and Ellen M Voorhees. | 2020. | Overview |     |     |     |     |     |
| ------- | --------------------- | ----- | -------- | --- | --- | --- | --- | --- |
ofthetrec2019deeplearningtrack. arXivpreprint Lewis,LedellWu,SergeyEdunov,DanqiChen,and
arXiv:2003.07820. Wen-tau Yih. 2020b. Dense passage retrieval for
|                                              |     |                |     | open-domain       | question | answering. | arXiv | preprint |
| -------------------------------------------- | --- | -------------- | --- | ----------------- | -------- | ---------- | ----- | -------- |
| FeitengFang,YuelinBai,ShiwenNi,MinYang,Xiao- |     |                |     | arXiv:2004.04906. |          |            |       |          |
| junChen,andRuifengXu.2024.                   |     | Enhancingnoise |     |                   |          |            |       |          |
robustnessofretrieval-augmentedlanguagemodels Seonghyeon Lee, Dongha Lee, Seongbo Jang, and
with adaptive adversarial training. In Proceedings HwanjoYu.2022. Towardinterpretablesemantictex-
of the 62nd Annual Meeting of the Association for tualsimilarityviaoptimaltransport-basedcontrastive
ComputationalLinguistics(Volume1: LongPapers), sentencelearning. arXivpreprintarXiv:2202.13196.
1837

DanLi,VikrantYadav,ZubairAfzal,andGeorgeTsat- HaoranWangandKaiShu.2023. Explainableclaim
saronis.2022. Unsuperviseddenseretrievalforscien- verificationviaknowledge-groundedreasoningwith
tificarticles. InProceedingsofthe2022Conference largelanguagemodels. InFindingsoftheAssocia-
onEmpiricalMethodsinNaturalLanguageProcess- tionforComputationalLinguistics: EMNLP2023,
ing: IndustryTrack,pages313–321. pages6288–6304,Singapore.AssociationforCom-
putationalLinguistics.
| Jianqiao | Li, Chunyuan | Li, Guoyin | Wang, |     | Hao Fu, |     |     |     |     |
| -------- | ------------ | ---------- | ----- | --- | ------- | --- | --- | --- | --- |
YuhchenLin,LiqunChen,YizheZhang,Chenyang ZihaoWang,JiahengDou,andYongZhang.2022. Un-
Tao,RuiyiZhang,WenlinWang,DinghanShen,Qian supervisedsentencetextualsimilaritywithcomposi-
Yang, and Lawrence Carin. 2020. Improving text tionalphrasesemantics. InProceedingsofthe29th
generationwithstudent-forcingoptimaltransport. In InternationalConferenceonComputationalLinguis-
| Proceedings | of the 2020 | Conference |     | on Empirical |     |     |     |     |     |
| ----------- | ----------- | ---------- | --- | ------------ | --- | --- | --- | --- | --- |
tics,pages4976–4995.
MethodsinNaturalLanguageProcessing(EMNLP),
pages9144–9156,Online.AssociationforComputa- Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai
tionalLinguistics. Zhang, and Yanghua Xiao. 2024. How easily do
irrelevantinputsskewtheresponsesoflargelanguage
ShesheraMysore,ArmanCohan,andTomHope.2021. models? Preprint,arXiv:2404.03302.
Multi-vectormodelswithtextualguidanceforfine-
grainedscientificdocumentsimilarity. arXivpreprint ShitaoXiao,ZhengLiu,PeitianZhang,NiklasMuen-
arXiv:2111.08366. nighoff,DefuLian,andJian-YunNie.2024. C-pack:
|     |     |     |     |     |     | Packed resources |     | for general chinese | embeddings. |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------------- | ----------- |
TheoOlausson,AlexGu,BenLipkin,CedegaoZhang,
Preprint,arXiv:2309.07597.
| Armando         | Solar-Lezama,               | Joshua | Tenenbaum, |     | and |     |     |     |     |
| --------------- | --------------------------- | ------ | ---------- | --- | --- | --- | --- | --- | --- |
| RogerLevy.2023. | LINC:Aneurosymbolicapproach |        |            |     |     |     |     |     |     |
LeeXiong,ChenyanXiong,YeLi,Kwok-FungTang,
forlogicalreasoningbycombininglanguagemodels
JialinLiu,PaulBennett,JunaidAhmed,andArnold
withfirst-orderlogicprovers. InProceedingsofthe Overwijk.2020. Approximatenearestneighborneg-
2023ConferenceonEmpiricalMethodsinNatural ative contrastive learning for dense text retrieval.
LanguageProcessing,pages5153–5176,Singapore. arXivpreprintarXiv:2007.00808.
AssociationforComputationalLinguistics.
JundongXu,HaoFei,LiangmingPan,QianLiu,Mong-
JielinQiu,JiachengZhu,MengdiXu,FranckDernon-
|     |     |     |     |     |     | LiLee,andWynneHsu.2024. |     | Faithfullogicalrea- |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | --- | ------------------- | --- |
court,TrungBui,ZhaowenWang,BoLi,DingZhao,
|                    |     |                           |     |     |     | soningviasymbolicchain-of-thought. |     |     | InProceed- |
| ------------------ | --- | ------------------------- | --- | --- | --- | ---------------------------------- | --- | --- | ---------- |
| andHailinJin.2023. |     | SCCS:Semantics-consistent |     |     |     |                                    |     |     |            |
ingsofthe62ndAnnualMeetingoftheAssociation
| cross-domain | summarization |     | via optimal | transport |     |                                      |     |     |         |
| ------------ | ------------- | --- | ----------- | --------- | --- | ------------------------------------ | --- | --- | ------- |
|              |               |     |             |           |     | forComputationalLinguistics(Volume1: |     |     | LongPa- |
alignment. InFindingsoftheAssociationforCom- pers),pages13326–13365,Bangkok,Thailand.As-
| putationalLinguistics: |     | ACL2023,pages1584–1601, |     |     |     |     |     |     |     |
| ---------------------- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- |
sociationforComputationalLinguistics.
Toronto,Canada.AssociationforComputationalLin-
| guistics. |     |     |     |     |     | PeilinYang,HuiFang,andJimmyLin.2017. |     |     | Anserini: |
| --------- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --------- |
Enablingtheuseofluceneforinformationretrieval
YingqiQu,YuchenDing,JingLiu,KaiLiu,Ruiyang
|            |           |         |       |     |     | research. | In Proceedings | of the 40th | international |
| ---------- | --------- | ------- | ----- | --- | --- | --------- | -------------- | ----------- | ------------- |
| Ren, Wayne | Xin Zhao, | Daxiang | Dong, | Hua | Wu, |           |                |             |               |
ACMSIGIRconferenceonresearchanddevelopment
| and Haifeng | Wang. | 2020. | Rocketqa: |     | An opti- |     |     |     |     |
| ----------- | ----- | ----- | --------- | --- | -------- | --- | --- | --- | --- |
ininformationretrieval,pages1253–1256.
mizedtrainingapproachtodensepassageretrieval
foropen-domainquestionanswering. arXivpreprint YuanYang,SihengXiong,AliPayani,EhsanShareghi,
arXiv:2010.08191. andFaramarzFekri.2024. Harnessingthepowerof
IevgenRedko,NicolasCourty,RémiFlamary,andDe- largelanguagemodelsfornaturallanguagetofirst-
|               |                                 |     |     |     |     | orderlogictranslation. |     | InProceedingsofthe62nd |     |
| ------------- | ------------------------------- | --- | --- | --- | --- | ---------------------- | --- | ---------------------- | --- |
| visTuia.2019. | Optimaltransportformulti-source |     |     |     |     |                        |     |                        |     |
AnnualMeetingoftheAssociationforComputational
| domainadaptationundertargetshift. |     |     |     | InThe22ndIn- |     |             |         |                  |             |
| --------------------------------- | --- | --- | --- | ------------ | --- | ----------- | ------- | ---------------- | ----------- |
|                                   |     |     |     |              |     | Linguistics | (Volume | 1: Long Papers), | pages 6942– |
ternationalConferenceonartificialintelligenceand
6959,Bangkok,Thailand.AssociationforComputa-
statistics,pages849–858.PMLR.
tionalLinguistics.
| StephenRobertson,HugoZaragoza,etal.2009. |     |     |     |     | The |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
KaiZhang,ChongyangTao,TaoShen,CanXu,Xiubo
| probabilistic | relevance | framework: |     | Bm25 | and be- |     |     |     |     |
| ------------- | --------- | ---------- | --- | ---- | ------- | --- | --- | --- | --- |
yond. FoundationsandTrends®inInformationRe- Geng, Binxing Jiao, and Daxin Jiang. 2023. Led:
trieval,3(4):333–389. Lexicon-enlighteneddenseretrieverforlarge-scale
|     |     |     |     |     |     | retrieval. | InProceedingsoftheACMWebConference |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | ---------------------------------- | --- | --- |
TaoShen,XiuboGeng,ChongyangTao,CanXu,Xiao- 2023,pages3203–3213.
longHuang,BinxingJiao,LinjunYang,andDaxin
ShuyingZhang,TianyuZhao,andTatsuyaKawahara.
| Jiang. | 2022. Lexmae: | Lexicon-bottlenecked |     |     | pre- |     |     |     |     |
| ------ | ------------- | -------------------- | --- | --- | ---- | --- | --- | --- | --- |
2020. Topic-relevantresponsegenerationusingopti-
| training | for large-scale | retrieval. |     | arXiv | preprint |     |     |     |     |
| -------- | --------------- | ---------- | --- | ----- | -------- | --- | --- | --- | --- |
arXiv:2208.14754. maltransportforanopen-domaindialogsystem. In
Proceedingsofthe28thInternationalConferenceon
N Thakur, N Reimers, A Rücklé, A Srivastava, and ComputationalLinguistics,pages4067–4077.
| IBeirGurevych.2021. |     | Aheterogenousbenchmark |     |     |     |     |     |     |     |
| ------------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
forzero-shotevaluationofinformationretrievalmod- ChenZhao,ChenyanXiong,JordanBoyd-Graber,and
els. arXivpreprintarXiv:2104.08663. Hal Daumé Iii. 2021. Distantly-supervised dense
1838

| retrieval enables          | open-domain | question answering |
| -------------------------- | ----------- | ------------------ |
| withoutevidenceannotation. |             | InProceedingsofthe |
2021ConferenceonEmpiricalMethodsinNatural
LanguageProcessing,pages9612–9622.
YanZhou,QingkaiFang,andYangFeng.2023. CMOT:
Cross-modalmixupviaoptimaltransportforspeech
| translation. | InProceedingsofthe61stAnnualMeet- |     |
| ------------ | --------------------------------- | --- |
ingoftheAssociationforComputationalLinguistics
| (Volume1: | LongPapers),pages7873–7887,Toronto, |     |
| --------- | ----------------------------------- | --- |
Canada.AssociationforComputationalLinguistics.
A FOLExamples
Table5providesanexampleoftheNL-query,FOL-
query,NL-document,andFOL-document. TheNL-
queryandNL-documentstemfromactualdatasets.
FOL-queryandFOL-documentareFOLformats
| of NL-queries | and NL-documents, | respectively, |
| ------------- | ----------------- | ------------- |
whicharegeneratedbyGPT-4ointhispaper.
1839

| NL-query  | Whatisconsideredabusinessexpenseonabusinesstrip? |                     |     |     |     |
| --------- | ------------------------------------------------ | ------------------- | --- | --- | --- |
| FOL-query | x(BusinessTrip(x)                                | BusinessExpense(x)) |     |     |     |
|           | ∀                                                | →                   |     |     |     |
I’mnotsayingIdon’tliketheideaofon-the-jobtrainingtoo,butyoucan’texpectthecompanytodo
that.Trainingworkersisnottheirjob-they’rebuildingsoftware.Perhapseducationalsystemsinthe
NL-documnet U.S.(ortheirstudents)shouldworryalittleaboutgettingmarketableskillsinexchangefortheirmassive
investmentineducation,ratherthangettingoutwiththousandsinstudentdebtandthencomplainingthat
theyaren’tqualifiedtodoanything.
|     |     | Like(i,onTheJobTraining) | Expect(company,Train(workers)) |     |     |
| --- | --- | ------------------------ | ------------------------------ | --- | --- |
|     | ¬   |                          | ∧¬                             |     |     |
Job(company,Train(workers))
| FOL-document |     |     | ¬   |     |     |
| ------------ | --- | --- | --- | --- | --- |
Job(company,Build(software)) x(EducationalSystems(x) Worry(x,MarketableSkills))
|     | ∧   |     | ∀   | →   |     |
| --- | --- | --- | --- | --- | --- |
Invest(educationalSystems,education) x(Student(x) StudentDebt(x)) Qualified(x,anything)
|     |     |     | ∧∃  | ∧   | →¬  |
| --- | --- | --- | --- | --- | --- |
Table5: AnexampleoftheNL-query,FOL-query,NL-documentandFOL-document.
1840

B Prompts
PromptofNL-FOLTranslationforQueries
Givensomequestion.Thetaskistoparsethesequestionsintofirst-orderlogicformulars.Thegrammarofthefirst-order
logicformularisdefinedasfollows:
1)logicalconjunctionofexpr1andexpr2:expr1 expr2
∧
2)logicaldisjunctionofexpr1andexpr2:expr1 expr2
∨
3)logicalexclusivedisjunctionofexpr1andexpr2:expr1 expr2
⊕
4)logicalnegationofexpr1: expr1
¬
5)expr1impliesexpr2:expr1 expr2
→
6)expr1ifandonlyifexpr2:expr1 expr2
↔
7)logicaluniversalquantification: x
∃
8)logicalexistentialquantification: x
∀
——
Hereisanexample:
Query:
Rinaiseitherapersonwhojokesaboutbeingaddictedtocaffeineorisunawarethatcaffeineisadrug.
IfRinaiseitherapersonwhojokesaboutbeingaddictedtocaffeineandapersonwhoisunawarethatcaffeineisadrug,
orneitherapersonwhojokesaboutbeingaddictedtocaffeinenorapersonwhoisunawarethatcaffeineisadrug,then
Rinajokesaboutbeingaddictedtocaffeineandregularlydrinkscoffee.
###
Predicates:
Drinks(x):::xregularlydrinkscoffee.
Jokes(x):::xjokesaboutbeingaddictedtocaffeine.
Unaware(x):::xisunawarethatcaffeineisadrug.
Conclusion:
Jokes(rina) Unaware(rina):::Rinaiseitherapersonwhojokesaboutbeingaddictedtocaffeineorisunawarethat
⊕
caffeineisadrug.
((Jokes(rina) Unaware(rina)) (Jokes(rina) Unaware(rina))) (Jokes(rina) Drinks(rina)):::IfRinaiseithera
∧ ⊕¬ ∨ → ∧
personwhojokesaboutbeingaddictedtocaffeineandapersonwhoisunawarethatcaffeineisadrug,orneithera
personwhojokesaboutbeingaddictedtocaffeinenorapersonwhoisunawarethatcaffeineisadrug,thenRinajokes
aboutbeingaddictedtocaffeineandregularlydrinkscoffee.
——
Hereisanexample:
Query:
MiroslavVenhodalovedmusic.
ACzechpersonwroteabookin1946.
NochoralconductorspecializedintheperformanceofRenaissance.
###
Predicates:
Czech(x):::xisaCzechperson.
ChoralConductor(x):::xisachoralconductor.
Author(x,y):::xistheauthorofy.
Book(x):::xisabook.
Specialize(x,y):::xspecializesiny.
Conclusion:
Love(miroslav,music):::MiroslavVenhodalovedmusic.
y x(Czech(x) Author(x,y) Book(y) Publish(y,year1946)):::ACzechpersonwroteabookin1946.
∃ ∃ ∧ ∧ ∧
x(ChoralConductor(x) Specialize(x,renaissance))::: Nochoralconductorspecializedintheperformanceof
¬∃ ∧
Renaissance.
——
Belowistheoneyouneedtotranslate:
Query:
%QUERY%
——-
1841

PromptofNL-FOLTranslationforDocuments
Givenadocument.Thetaskistoparsethedocumentintofirst-orderlogicformulars.Thegrammarofthefirst-order
logicformularisdefinedasfollows:
| 1)logicalconjunctionofexpr1andexpr2:expr1          |     |         | expr2   |     |     |
| -------------------------------------------------- | --- | ------- | ------- | --- | --- |
| 2)logicaldisjunctionofexpr1andexpr2:expr1          |     |         | ∧ expr2 |     |     |
| 3)logicalexclusivedisjunctionofexpr1andexpr2:expr1 |     |         | ∨ expr2 |     |     |
| 4)logicalnegationofexpr1:                          |     | expr1   | ⊕       |     |     |
| 5)expr1impliesexpr2:expr1                          |     | ¬ expr2 |         |     |     |
| 6)expr1ifandonlyifexpr2:expr1                      |     | → expr2 |         |     |     |
| 7)logicaluniversalquantification:                  |     | ↔ x     |         |     |     |
| 8)logicalexistentialquantification:                |     | ∃ x     |         |     |     |
| ——                                                 |     | ∀       |         |     |     |
Hereisanexample:
Document:
Allpeoplewhoregularlydrinkcoffeearedependentoncaffeine. Peopleeitherregularlydrinkcoffeeorjokeabout
beingaddictedtocaffeine.Noonewhojokesaboutbeingaddictedtocaffeineisunawarethatcaffeineisadrug.Rinais
eitherastudentandunawarethatcaffeineisadrug,orneitherastudentnorunawarethatcaffeineisadrug.IfRinais
notapersondependentoncaffeineandastudent,thenRinaiseitherapersondependentoncaffeineandastudent,or
neitherapersondependentoncaffeinenorastudent.
###
Predicates:
Dependent(x):::xisapersondependentoncaffeine.
Drinks(x):::xregularlydrinkscoffee.
Jokes(x):::xjokesaboutbeingaddictedtocaffeine.
Unaware(x):::xisunawarethatcaffeineisadrug.
Student(x):::xisastudent.
Conclusion:
x(Drinks(x) Dependent(x)):::Allpeoplewhoregularlydrinkcoffeearedependentoncaffeine.
| ∀   | →   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
x(Drinks(x) Jokes(x)):::Peopleeitherregularlydrinkcoffeeorjokeaboutbeingaddictedtocaffeine.
| ∀   | ⊕   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
x(Jokes(x) Unaware(x)):::Noonewhojokesaboutbeingaddictedtocaffeineisunawarethatcaffeineisadrug.
| ∀   | →¬  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
(Student(rina) Unaware(rina)) (Student(rina) Unaware(rina))::: Rinaiseitherastudentandunawarethat
|     | ∧   | ⊕¬  | ∨   |     |     |
| --- | --- | --- | --- | --- | --- |
caffeineisadrug,orneitherastudentnorunawarethatcaffeineisadrug.
(Dependent(rina) Student(rina)) (Dependent(rina) Student(rina)) (Dependent(rina) Student(rina)):::If
| ¬   | ∧   | →   | ∧   | ⊕¬  | ∨   |
| --- | --- | --- | --- | --- | --- |
Rinaisnotapersondependentoncaffeineandastudent,thenRinaiseitherapersondependentoncaffeineanda
student,orneitherapersondependentoncaffeinenorastudent.
——
Hereisanexample:
Document:
MiroslavVenhodawasaCzechchoralconductorwhospecializedintheperformanceofRenaissanceandBaroque
music.Anychoralconductorisamusician.Somemusicianslovemusic.MiroslavVenhodapublishedabookin1946
calledMethodofStudyingGregorianChant.
###
Predicates:
Czech(x):::xisaCzechperson.
ChoralConductor(x):::xisachoralconductor.
Musician(x):::xisamusician.
Love(x,y):::xlovesy.
Author(x,y):::xistheauthorofy.
Book(x):::xisabook.
Publish(x,y):::xispublishedinyeary.
Specialize(x,y):::xspecializesiny.
Conclusion:
Czech(miroslav) ChoralConductor(miroslav) Specialize(miroslav,renaissance) Specialize(miroslav,baroque):::
|     | ∧   |     | ∧   | ∧   |     |
| --- | --- | --- | --- | --- | --- |
MiroslavVenhodawasaCzechchoralconductorwhospecializedintheperformanceofRenaissanceandBaroque
music.
| x(ChoralConductor(x) |                                          | Musician(x)):::Anychoralconductorisamusician. |     |     |     |
| -------------------- | ---------------------------------------- | --------------------------------------------- | --- | --- | --- |
| ∃                    | →                                        |                                               |     |     |     |
| x(Musician(x)        | Love(x,music)):::Somemusicianslovemusic. |                                               |     |     |     |
| ∀                    | ∧                                        |                                               |     |     |     |
Book(methodOfStudyingGregorianChant) Author(miroslav, methodOfStudyingGregorianChant) Pub-
∧ ∧
lish(methodOfStudyingGregorianChant,year1946):::MiroslavVenhodapublishedabookin1946calledMethodof
StudyingGregorianChant.
——
Belowistheoneyouneedtotranslate:
Document:
%DOCUMENT%
1842

PromptofQueryGenerationforA-a
Givenanexampleininformationretrievaltasks.Werefertothequeryasanegative-constraintquery.Thequerymatches
formulationA-a. AdenotesGinsberg’sworksandadenotes’Howl’. ThepositivedocumentmentionsGinsberg’s
worksbutdoesnotmention’Howl’.ThenegativedocumentmentionsGinsberg’sworksand’Howl’.Pleaseprovidea
querybasedonthepositiveandnegativedocumentsprovided.
——
EXAMPLE
Positivedocument:
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainsteverythingfromtheVietnamWartotheWaron
Drugs.Hispoem""SeptemberonJessoreRoad"",callingattentiontotheplightofBangladeshirefugees,exemplifies
whattheliterarycriticHelenVendlerdescribedasGinsberg’stirelesspersistenceinprotestingagainst""imperialpolitics,
andpersecutionofthepowerless.""Hiscollection""TheFallofAmerica""sharedtheannualU.S.NationalBook
AwardforPoetryin1974.In1979hereceivedtheNationalArtsClubgoldmedalandwasinductedintotheAmerican
AcademyandInstituteofArtsandLetters.GinsbergwasaPulitzer.bygambling,andthecostofsecondaryeducation
forPoe. HeattendedtheUniversityofVirginiabutleftafterayearduetolackofmoney. PoequarreledwithAllan
overthefundsforhiseducationandenlistedintheArmyin1827underanassumedname.Itwasatthistimethathis
publishingcareerbegan,albeithumbly,withtheanonymouscollection""TamerlaneandOtherPoems""(1827),credited
onlyto""aBostonian"".WiththedeathofFrancesAllanin1829,PoeandAllanreachedatemporaryrapprochement.
However,PoelaterfailedasanofficercadetatWestPoint,declaring...
Negativedocument:
KerouacandWilliamS.Burroughs.Ginsbergisbestknownforhispoem""Howl"",inwhichhedenouncedwhathe
sawasthedestructiveforcesofcapitalismandconformityintheUnitedStates.In1956,""Howl""wasseizedbySan
FranciscopoliceandUSCustoms.In1957,itattractedwidespreadpublicitywhenitbecamethesubjectofanobscenity
trial,asitdescribedheterosexualandhomosexualsexatatimewhensodomylawsmadehomosexualactsacrime
ineveryU.S.state. ""Howl""reflectedGinsberg’sownhomosexualityandhisrelationshipswithanumberofmen,
includingPeterOrlovsky,hislifelongpartner.
Query:
IntroduceAllenGinsberg’sworks,butdonotmention’Howl’.
——
Positivedocument:%POSITIVEDOCUMENT%
Negativedocument:%NEGATIVEDOCUMENT%
——
Belowqueryistheoneyouneedtogenerate,whichmakesignificantchangestothequerystyle.
Query:
%QUERY%
1843

PromptofQueryGeneration(A-a) B
∪
Givenanexampleininformationretrievaltasks.Werefertothequeryasanegative-constraintquery.Thequerymatches
formulation(A-a) B.AdenotesAllenGinsberg’sworks,adenotes’Howl’,andBdenotesEdgarAllanPoe’sworks.
∪
ThepositivedocumentmentionsAllenGinsberg’sandEdgarAllanPoe’sworksbutdoesnotmention’Howl’. The
negativedocumentmentionsAllenGinsberg’sworks,EdgarAllanPoe’sworksand’Howl’. Pleaseprovideaquery
basedonthepositiveandnegativedocumentsprovided.
——
EXAMPLE
Positivedocument:
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainsteverythingfromtheVietnamWartotheWaron
Drugs.Hispoem""SeptemberonJessoreRoad"",callingattentiontotheplightofBangladeshirefugees,exemplifies
whattheliterarycriticHelenVendlerdescribedasGinsberg’stirelesspersistenceinprotestingagainst""imperialpolitics,
andpersecutionofthepowerless.""Hiscollection""TheFallofAmerica""sharedtheannualU.S.NationalBook
AwardforPoetryin1974.In1979hereceivedtheNationalArtsClubgoldmedalandwasinductedintotheAmerican
AcademyandInstituteofArtsandLetters.GinsbergwasaPulitzer.bygambling,andthecostofsecondaryeducation
forPoe. HeattendedtheUniversityofVirginiabutleftafterayearduetolackofmoney. PoequarreledwithAllan
overthefundsforhiseducationandenlistedintheArmyin1827underanassumedname.Itwasatthistimethathis
publishingcareerbegan,albeithumbly,withtheanonymouscollection""TamerlaneandOtherPoems""(1827),credited
onlyto""aBostonian"".WiththedeathofFrancesAllanin1829,PoeandAllanreachedatemporaryrapprochement.
However,PoelaterfailedasanofficercadetatWestPoint,declaring...
Negativedocument:
KerouacandWilliamS.Burroughs.Ginsbergisbestknownforhispoem""Howl"",inwhichhedenouncedwhathe
sawasthedestructiveforcesofcapitalismandconformityintheUnitedStates.In1956,""Howl""wasseizedbySan
FranciscopoliceandUSCustoms.In1957,itattractedwidespreadpublicitywhenitbecamethesubjectofanobscenity
trial,asitdescribedheterosexualandhomosexualsexatatimewhensodomylawsmadehomosexualactsacrime
ineveryU.S.state. ""Howl""reflectedGinsberg’sownhomosexualityandhisrelationshipswithanumberofmen,
includingPeterOrlovsky,hislifelongpartner.bygambling,andthecostofsecondaryeducationforPoe.Heattended
theUniversityofVirginiabutleftafterayearduetolackofmoney. PoequarreledwithAllanoverthefundsforhis
educationandenlistedintheArmyin1827underanassumedname.Itwasatthistimethathispublishingcareerbegan,
albeithumbly,withtheanonymouscollection""TamerlaneandOtherPoems""(1827),creditedonlyto""aBostonian"".
WiththedeathofFrancesAllanin1829,PoeandAllanreachedatemporaryrapprochement.However,Poelaterfailed
asanofficercadetatWestPoint,declaring...
Query:
WhatthemesareexpressedinAllenGinsberg’sworks(excluding"Howl")?IsthereanysimilaritybetweenEdgarAllan
Poe’sworksandtheirs?
——
Positivedocument:%POSITIVEDOCUMENT%
Negativedocument:%NEGATIVEDOCUMENT%
——
Belowqueryistheoneyouneedtogenerate,whichmakesignificantchangestothequerystyle.
Query:
%QUERY%
1844

PromptofQueryGeneration(A-a) (B-b)
∪
Givenanexampleininformationretrievaltasks.Werefertothequeryasanegative-constraintquery.Thequerymatches
formulation(A-a) (B-b).AdenotesGinsberg’sworks,adenotes’Howl’,BdenotesPoe’sworksandbdenotes
∪
’TheRaven’. ThepositivedocumentmentionsGinsberg’sandPoe’sworksbutdoesnotmention’Howl’and’The
Raven’. Thenegativedocument1mentionsGinsberg’sworks,Poe’sworks’and’Howl’. Thenegativedocument2
mentionsGinsberg’sworks,Poe’sworks’and’TheRaven’.Thenegativedocument3mentionsGinsberg’sworks,Poe’s
works’,’Howl’and’TheRaven’.Pleaseprovideaquerybasedonthepositiveandnegativedocumentsprovided.
——
EXAMPLE
Positivedocument:
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainsteverythingfromtheVietnamWartotheWaron
Drugs.Hispoem""SeptemberonJessoreRoad"",callingattentiontotheplightofBangladeshirefugees,exemplifies
whattheliterarycriticHelenVendlerdescribedasGinsberg’stirelesspersistenceinprotestingagainst""imperialpolitics,
andpersecutionofthepowerless.""Hiscollection""TheFallofAmerica""sharedtheannualU.S.NationalBook
AwardforPoetryin1974.In1979hereceivedtheNationalArtsClubgoldmedalandwasinductedintotheAmerican
AcademyandInstituteofArtsandLetters. GinsbergwasaPulitzer. producehisownjournal""ThePenn""(later
renamed""TheStylus""),thoughhediedbeforeitcouldbeproduced.bygambling,andthecostofsecondaryeducation
forPoe. HeattendedtheUniversityofVirginiabutleftafterayearduetolackofmoney. PoequarreledwithAllan
overthefundsforhiseducationandenlistedintheArmyin1827underanassumedname.Itwasatthistimethathis
publishingcareerbegan,albeithumbly,withtheanonymouscollection""TamerlaneandOtherPoems""(1827),credited
onlyto""aBostonian"".WiththedeathofFrancesAllanin1829,PoeandAllanreachedatemporaryrapprochement.
However,PoelaterfailedasanofficercadetatWestPoint,declaring...
Negativedocument1:
KerouacandWilliamS.Burroughs.Ginsbergisbestknownforhispoem""Howl"",inwhichhedenouncedwhathe
sawasthedestructiveforcesofcapitalismandconformityintheUnitedStates.In1956,""Howl""wasseizedbySan
FranciscopoliceandUSCustoms.In1957,itattractedwidespreadpublicitywhenitbecamethesubjectofanobscenity
trial,asitdescribedheterosexualandhomosexualsexatatimewhensodomylawsmadehomosexualactsacrime
ineveryU.S.state. ""Howl""reflectedGinsberg’sownhomosexualityandhisrelationshipswithanumberofmen,
includingPeterOrlovsky...
Negativedocument2:
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainsteverythingfromtheVietnamWartotheWaron
Drugs.Hispoem""SeptemberonJessoreRoad"",callingattentiontotheplightofBangladeshirefugees,exemplifies
whattheliterarycriticHelenVendlerdescribedasGinsberg’stirelesspersistenceinprotestingagainst""imperialpolitics,
andpersecutionofthepowerless.""Hiscollection""TheFallofAmerica""sharedtheannualU.S.NationalBookAward
forPoetryin1974.In1979hereceivedtheNationalArtsClubgoldmedalandwasinductedintotheAmericanAcademy
andInstituteofArtsandLetters.GinsbergwasaPulitzer.producehisownjournal""ThePenn""(laterrenamed""The
Stylus""),thoughhediedbeforeitcouldbeproduced.afirmwishtobeapoetandwriter...
Negativedocument3:
KerouacandWilliamS.Burroughs.Ginsbergisbestknownforhispoem""Howl"",inwhichhedenouncedwhathe
sawasthedestructiveforcesofcapitalismandconformityintheUnitedStates.In1956,""Howl""wasseizedbySan
FranciscopoliceandUSCustoms.In1957,itattractedwidespreadpublicitywhenitbecamethesubjectofanobscenity
trial,asitdescribedheterosexualandhomosexualsexatatimewhensodomylawsmadehomosexualactsacrime
ineveryU.S.state. ""Howl""reflectedGinsberg’sownhomosexualityandhisrelationshipswithanumberofmen,
includingPeterOrlovsky...
Query:
WhatarethesimilaritiesbetweenGinsberg’sworks(excluding’Howl’)andPoe’sworks(excluding’TheRaven’)?
——
Positivedocument:%POSITIVEDOCUMENT%
Negativedocument1:%NEGATIVEDOCUMENT1%
Negativedocument2:%NEGATIVEDOCUMENT2%
Negativedocument3:%NEGATIVEDOCUMENT3%
——
Belowqueryistheoneyouneedtogenerate,whichmakesignificantchangestothequerystyle.
Query:
%QUERY%
1845

C DataCollectionandSnippet
WeusetheintroductorypassagesfromWikipedia
dumpasthecorpus,astheyareusuallyhigh-quality
and contain most of the key information. We re-
questthreecarefullyselectedexperiencedannota-
torstofilterpassagesfromWikipedia. Forqueries
offormulationA-aand(A-a) B,eachpositive
∪
and negative document corresponding to a query
is composed of one passage from the Wikipedia
dump,respectively. However,duetothecomplex-
ity of formulation (A - a) (B - b), we merge
∪
twopassagesthatbelongtoonetopicasapositive
document,andthetwomergedpassagesarehighly
relevant. Negativedocumentsforformulation(A-
a) (B-b)arealsoobtainedinthisway. Thenwe
∪
prompt GPT-4o to generate queries based on the
positiveandnegativedocuments. Weensurethat
the queries are as style-diverse as possible. That
is,wedonotjustperformentityreplacement,but
paymoreattentiontothediversityofquerymode.
For example, there are three queries expressing
negativeconstraintsforformulation A-a:
1. Investigate the role of nature in Walden, ex-
cludingThoreau’scritiqueofsociety.
2. IntroducetheworksofEmilyDickinson,but
donotmention’BecauseIcouldnotstopfor
Death’.
3. WithoutreferencingVictorFrankenstein’suse
of scientific knowledge, examine the role of
technologyinFrankenstein.
Finally,annotatorsalsoselectseveralirrelevantpas-
sageswithqueriestofillintothecorpus.
Table 6 introduce snippets of NegConstraint
dataset. Entities marked in red and green denote
entitiesinnegative-constraintconditions. Forfor-
mulation A - a, the negative document mentions
"Howl". Forformulation (A-a) B,thenegative
∪
document mentions "Howl". For formulation (A
- a) (B - b), the negative document 1 mentions
∪
"Howl", the negative document 2 mentions "The
Raven", and the negative document 3 mentions
"Howl"and"TheRaven".
1846

Query IntroduceAllenGinsberg’sworks,butdonotmention’Howl’.
...Hispoem’SeptemberonJessoreRoad’, callingattentiontotheplight
ofBangladeshirefugees,exemplifieswhattheliterarycriticHelenVendler
Postivedocument
describedasGinsberg’stirelesspersistenceinprotestingagainst""imperial
A-a
politics,andpersecutionofthepowerless...
In1956,’Howl’wasseizedbySanFranciscopoliceandUSCustoms...’Howl’
Negativedocument reflectedGinsberg’sownhomosexualityandhisrelationshipswithanumber
ofmen,includingPeterOrlovsky,hislifelongpartner...
WhatthemesareexpressedinAllenGinsberg’sworksotherthan’Howl’and
Query
EdgarAllanPoe’sworks?
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainstevery-
thingfromtheVietnamWartotheWaronDrugs. Hispoem""September
onJessoreRoad"",callingattentiontotheplightofBangladeshirefugees,
Postivedocument
exemplifieswhattheliterarycriticHelenVendlerdescribedasGinsberg’s
(A-a) B
∪ tirelesspersistenceinprotestingagainst""imperialpolitics,andpersecution
ofthepowerless...
KerouacandWilliamS.Burroughs. Ginsbergisbestknownforhispoem
’Howl’, in which he denounced what he saw as the destructive forces of
Negativedocument
capitalismandconformityintheUnitedStates.In1956,’Howl’wasseized
bySanFranciscopoliceandUSCustoms....
WhatthemesdoAllenGinsberg’sworksotherthan’Howl’andEdgarAllan
Query
Poe’sworksotherthan’TheRaven’express?
Ginsbergtookpartindecadesofnon-violentpoliticalprotestagainstevery-
thingfromtheVietnamWartotheWaronDrugs...,Hiscollection’TheFall
ofAmerica’sharedtheannualU.S.NationalBookAwardforPoetryin1974.
In1979hereceivedtheNationalArtsClubgoldmedalandwasinductedinto
Postivedocument
theAmericanAcademyandInstituteofArtsandLetters...,creditedonlyto’a
Bostonian’.WiththedeathofFrancesAllanin1829,PoeandAllanreached
atemporaryrapprochement.However,Poelaterfailedasanofficercadetat
WestPoint,declaring...
KerouacandWilliamS.Burroughs. Ginsbergisbestknownforhispoem
’Howl’, in which he denounced what he saw as the destructive forces of
capitalismandconformityintheUnitedStates...’Howl’reflectedGinsberg’s
(A-a) (B-b) Negativedocument1
∪ ownhomosexualityandhisrelationshipswithanumberofmen,including
PeterOrlovsky,hislifelongpartner.bygambling,andthecostofsecondary
educationforPoe...
...InJanuary1845,Poepublishedhispoem’TheRaven’toinstantsuccess.
Negativedocument2 Hiswifediedoftuberculosistwoyearsafteritspublication.Foryears,hehad
beenplanningto...
In 1956, ’Howl’ was seized by San Francisco police and US Customs...
’Howl’reflectedGinsberg’sownhomosexualityandhisrelationshipswitha
numberofmen,includingPeterOrlovsky,hislifelongpartner. afirmwish
Negativedocument3 tobeapoetandwriter,andheultimatelypartedwayswithJohnAllan...In
January1845,Poepublishedhispoem’TheRaven’toinstantsuccess.His
wifediedoftuberculosistwoyearsafteritspublication. Foryears,hehad
beenplanningto
Table6: SnippetsofNegConstraintdataset. Entitiesmarkedinredandgreendenotenegative-constraintconditions.
1847
