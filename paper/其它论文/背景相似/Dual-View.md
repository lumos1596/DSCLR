Dual-View Training for Instruction-Following Information Retrieval
QingchengZeng†,*,PuxuanYu†,AmanMehta†,FuhengZhao†,RajhansSamdani†
∗NorthwesternUniversity†SnowflakeInc.
Abstract
Original Instruction Positive
Query
|     |     |     |     |     |     | Which type of volcano eruption has not been  |     |     | New Instruction Negative |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | ------------------------ | --- | --- |
seen?
Instruction-followinginformationretrieval(IF- Subglacial Volcanoes: Unseen Eruptions
6202 rpA 02  ]RI.sc[  1v54881.4062:viXra Subglacial volcanoes, which erupt beneath ice
IR)studiesretrievalsystemsthatmustnotonly Original Instruction s h e e t s   o r  g l a c ie r s ,  h a v e   n o t  b e e n   d ir e c tl y
|     |     |     |     |     |     |     |     |     | obs | e r v e d   e r u p t in g .  T | h e s e   v o l c a n o e s  a r e   fo r m ed  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------- | ----------------------------------------------- |
find documents relevant to a query, but also V o l c a n o e s   a r e   c la s s i fi e d   in t o   d if f e r e n t  t y p e s   when ....
|     |     |     |     |     |     | b a s e d   o n   t h e | ir   s h a p e   a n d   e r u | p t i o n   st y l e .  A   |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | ------------------------------ | --------------------------- | --- | --- | --- |
document is relevant if it describes a specific
| obeyexplicituserconstraintssuchasrequired |             |     |           |              |     | type of volcano that has not been directly   |     |     |                               |     |     |
| ----------------------------------------- | ----------- | --- | --------- | ------------ | --- | -------------------------------------------- | --- | --- | ----------------------------- | --- | --- |
|                                           |             |     |           |              |     | observed erupting, and provides information  |     |     | Original Instruction Negative |     |     |
| attributes,                               | exclusions, |     | or output | preferences. |     | about its formation or characteristics.      |     |     |                               |     |     |
New Instruction Positive
|     |     |     |     |     |     | New Instruction |     |     |     | Impact of Volcanic Eruptions on Climate |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --------------------------------------- | --- |
However, most retrievers are trained primar- Volcanic eruptions can have significant impacts
|     |     |     |     |     |     | Select documents that discuss the broader  |     |     | on the climate ... e.g. the eruption of Mount  |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | ---------------------------------------------- | --- | --- |
ilyforsemanticrelevanceandoftenfailtodis- environmental impacts or research insights  Tambora ... there are some types of eruptions
|     |     |     |     |     |     | associated with unwitnessed eruption types,  |     |     | that have not been witnessed first hand, which  |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | ----------------------------------------------- | --- | --- |
tinguishdocumentsthatmatchthetopicfrom strictly excluding descriptions of the volcano's  could provide further insights...
physical formation or structural classification.
| thosethatsatisfytheinstruction. |      |           |          | Wepropose |     |          |                                        |     |     |     |     |
| ------------------------------- | ---- | --------- | -------- | --------- | --- | -------- | -------------------------------------- | --- | --- | --- | --- |
| a dual-view                     | data | synthesis | strategy | based     | on  |          |                                        |     |     |     |     |
|                                 |      |           |          |           |     | Figure1: | Wesynthesizenewinstructionsthatreverse |     |     |     |     |
polarity reversal: given a query, a document therelevancepolarityofexistingdocuments,creating
thatisrelevantundertheinstruction,andahard challengingsamplesthatsharpentheretriever’ssensi-
| negative | that matches |     | the query | but | violates |     |     |     |     |     |     |
| -------- | ------------ | --- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- |
tivitytoinstructionalnuances.
theinstruction,wepromptanLLMtogenerate
acomplementaryinstructionunderwhichthe
| twodocumentsswaprelevancelabels. |     |     |     |     | Bypre- |          |           |      |         |              |     |
| -------------------------------- | --- | --- | --- | --- | ------ | -------- | --------- | ---- | ------- | ------------ | --- |
|                                  |     |     |     |     |        | relevant | documents | must | discuss | a particular | as- |
sentingthesamedocumentpairundercomple-
pect,bewritteninacertainstyle,orsatisfylength
mentaryinstructionsthatinverttheirrelevance
|               |          |        |           |     |           | requirements. |          | This paradigm |     | rigorously | tests the    |
| ------------- | -------- | ------ | --------- | --- | --------- | ------------- | -------- | ------------- | --- | ---------- | ------------ |
| labels, the   | training | signal | forces    | the | retriever |               |          |               |     |            |              |
|               |          |        |           |     |           | capacity      | of dense | retrievers    | to  | adapt      | their behav- |
| to reconsider | the      | same   | candidate | set | through   |               |          |               |     |            |              |
the instruction, rather than relying on fixed iorbasedondynamicin-contextdirectives,going
beyondstaticnotionsofrelevance.
| topical cues. | On  | a 305M-parameter |     |     | encoder, |     |     |     |     |     |     |
| ------------- | --- | ---------------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
ourmethodimprovesperformanceontheFol- Despite the growing number of instruction-
lowIRbenchmarkby45%,surpassinggeneral- awareretrievers,criticallimitationspersist. Weller
purposeembeddingmodelsofcomparableor
|               |         |     |              |     |         | et al. (2025a) | conducted |     | a systematic |     | evaluation |
| ------------- | ------- | --- | ------------ | --- | ------- | -------------- | --------- | --- | ------------ | --- | ---------- |
| larger scale. | Through |     | head-to-head |     | compar- |                |           |     |              |     |            |
usinghuman-annotatedinstructionsthatfundamen-
isonsatmatcheddatabudgets,wefurthershow
|     |     |     |     |     |     | tally alter | relevance | definitions. |     | By  | quantifying |
| --- | --- | --- | --- | --- | --- | ----------- | --------- | ------------ | --- | --- | ----------- |
thatdatadiversityandinstructionsupervision
sensitivitytoinstructionchangeswiththep-MRR
| play complementary |     |     | roles: | the former | pre- |     |     |     |     |     |     |
| ------------------ | --- | --- | ------ | ---------- | ---- | --- | --- | --- | --- | --- | --- |
metric,whichmeasureswhetheraretrieverranks
servesgeneralretrievalquality,whilethelatter
improvesinstructionsensitivity. Theseresults the preferred document higher when the instruc-
highlightthevalueoftargeteddatasynthesisfor tionchanges,theirfindingsrevealthatmostcurrent
buildingretrievalsystemsthatarebothbroadly
modelsfailtointernalizedetailedrelevancecriteria,
capableandinstruction-aware. relyinginsteadonsuperficialquery-documentsim-
ilarityandlargelyignoringthespecificconstraints
1 Introduction
imposedbyinstructions.
Instruction-followinginformationretrievalextends
Toaddressthis,Welleretal.(2025b)introduced
traditionalsemanticmatchingbyrequiringsystems a training paradigm centered on instruction neg-
toadheretobothaqueryandexplicituser-defined
|     |     |     |     |     |     | atives, documents |     | that | are semantically |     | relevant |
| --- | --- | --- | --- | --- | --- | ----------------- | --- | ---- | ---------------- | --- | -------- |
constraintsthatspecifyrelevancecriteria(Suetal.,
tothequerybutbecomeirrelevantonceaspecific
2023; Weller et al., 2025a). For instance, a user instructionisapplied. Whiletheirresultsdemon-
mightnotonlysubmitaquerybutalsospecifythat
stratetheeffectivenessofinstructionnegativesover
*WorkdoneduringinternshipatSnowflakeInc. standardhardnegatives,thesenegativescarryad-
1

ditional untapped potential: each one implicitly For datapoints augmented with our method, the
definesacomplementaryinstructionunderwhich training batch contains both the original and the
it becomes the relevant document. In this paper, polarity-reversedview. Themodelmusttherefore
,D+)
weexploitthisobservationbypromptinganLLM learn to assign high similarity to (q ⊕I
orig
tosynthesizesuchcomplementaryinstructionsthat and(q⊕I ,D−)simultaneously,whilepushing
new
reverse the relevance polarity of existing docu- thereversedassignmentsapart. Thisdualobjective
ment pairs (Figure 1). The same documents are directlypenalizesinstruction-agnosticrepresenta-
thusrepurposedundertwocomplementaryviews, tions: no single query encoding can satisfy both
compelling the retriever to attend to fine-grained constraints unless it genuinely conditions on the
instructionaldifferencesratherthansurface-level semanticcontentoftheinstruction,sinceq⊕I
orig
query-documentsimilarity. andq⊕I mustretrieveoppositedocuments.
new
| We make  | three             | contributions. |           | (1) We   | propose |                    |     |     |                     |     |     |     |
| -------- | ----------------- | -------------- | --------- | -------- | ------- | ------------------ | --- | --- | ------------------- | --- | --- | --- |
|          |                   |                |           |          |         | DataSynthesisSetup |     |     | WeemployQwen3-Next- |     |     |     |
| a simple | polarity-reversal |                | synthesis | strategy | that    |                    |     |     |                     |     |     |     |
80B-A3B-Instruct(QwenTeam,2025)astheback-
| improves | FollowIR | p-MRR | by  | 45% | on a 305M- |          |     |          |           |           |     |     |
| -------- | -------- | ----- | --- | --- | ---------- | -------- | --- | -------- | --------- | --------- | --- | --- |
|          |          |       |     |     |            | bone LLM | for | our data | synthesis | pipeline. |     | We  |
parameterencoder,surpassinggeneral-purposeem-
|                                         |     |     |     |     |     | construct | our | seed dataset |     | by selecting | instances |     |
| --------------------------------------- | --- | --- | --- | --- | --- | --------- | --- | ------------ | --- | ------------ | --------- | --- |
| beddingmodelsofcomparableorlargerscale. |     |     |     |     | (2) |           |     |              |     |              |           |     |
fromthepromptrieverdatathatcontainatleastone
| Through       | head-to-head |             | comparisons   |     | at matched |                                  |     |     |     |                |     |     |
| ------------- | ------------ | ----------- | ------------- | --- | ---------- | -------------------------------- | --- | --- | --- | -------------- | --- | --- |
|               |              |             |               |     |            | pre-existinginstructionnegative. |     |     |     | Foreachofthese |     |     |
| data budgets, |              | we identify | a fundamental |     | tension    |                                  |     |     |     |                |     |     |
datapoints,wegenerateanewinstructionthatre-
| in IF training: |     | data diversity |     | sustains | general re- |            |       |        |          |          |     |         |
| --------------- | --- | -------------- | --- | -------- | ----------- | ---------- | ----- | ------ | -------- | -------- | --- | ------- |
|                 |     |                |     |          |             | verses the | roles | of the | positive | document |     | and the |
trievalqualitywhileinstructionsupervisiondrives
|     |     |     |     |     |     | instructionnegative. |     | Thisprocessyieldsonecom- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------- | --- | ------------------------ | --- | --- | --- | --- |
IFcapability,butsupplementingwithnon-instruct
|     |     |     |     |     |     | plementary | training | sample |     | per original | instance. |     |
| --- | --- | --- | --- | --- | --- | ---------- | -------- | ------ | --- | ------------ | --------- | --- |
datamaydilutetheinstructionsignalandhurtIF
|              |     |                               |     |     |     | In our controlled |     | experiments, |         | DV  | samples      | sub- |
| ------------ | --- | ----------------------------- | --- | --- | --- | ----------------- | --- | ------------ | ------- | --- | ------------ | ---- |
| performance. |     | (3)Weshowthatdual-viewsynthe- |     |     |     |                   |     |              |         |     |              |      |
|              |     |                               |     |     |     | stitute for       | an  | equal-sized  | portion | of  | the training |      |
sisresolvesthistension,simultaneouslyimproving
|     |     |     |     |     |     | setratherthanbeingaddedontop, |     |     |     |     | enablingsize- |     |
| --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | ------------- | --- |
instructionsensitivityandgeneralretrievalatequal
|             |                                  |     |     |     |     | matchedcomparisonsacrossallconfigurations. |     |     |     |     |     | Af- |
| ----------- | -------------------------------- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
| databudget. | Allfindingsarevalidatedacrosstwo |     |     |     |     |                                            |     |     |     |     |     |     |
tersynthesis,oneannotatormanuallychecked100
encoderbackbones.
datapointsandconfirmedthatover99%oftheDV
|     |     |     |     |     |     | instructionsareusable. |     |     | Thus,noadditionalfilter- |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | ------------------------ | --- | --- | --- |
2 Methodology
|     |     |     |     |     |     | ingwasconducted. |     | Thespecificprompttemplate |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------------------- | --- | --- | --- | --- |
usedforthisgenerationisprovidedinAppendixA.
AsillustratedinFigure1,ourapproachleverages
LLMstogeneratecomplementaryinstructionsthat
3 ExperimentalSetup
invertthegroundtruthlabelsforafixedsetofdoc-
uments. Formally,givenaqueryq,apositivedoc- Backbone Models We adopt gte-multilingual-
umentD+,andaninstructionnegativedocument mlm-base(Zhangetal.,2024)(305Mparameters)
D− underanoriginalinstructionI ,weprompt as our primary encoder, initialized from our own
orig
theLLMtosynthesizeanewinstructionI . The contrastivelypretrainedcheckpointtrainedon1.41
new
generationisconstrainedsuchthatI new mustbese- billion unsupervised query-document pairs from
manticallycoherentwithq butsufficientlydistinct C4 (Raffel et al., 2020), mC4 (Habernal et al.,
from I so that D− becomes the relevant doc- 2016),CCNews,andmultilingualWikipedia. To
orig
D+
ument (positive) and becomes an instruction assess cross-backbone generalizability, we addi-
tionallytrainbge-m3-retromae(Chenetal.,2024)
| negative. | Thiscreatesa“dual-view”trainingsce- |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
nariowheretherelevanceofadocumentdepends under the same data configurations and report re-
| entirely | on the | specific | constraints | of  | the instruc- | sultsinTable2. |     |     |     |     |     |     |
| -------- | ------ | -------- | ----------- | --- | ------------ | -------------- | --- | --- | --- | --- | --- | --- |
tion,notonitstextorthequeryalone. Byrequiring • Ins-orig: 480kinstructsamplesfromtheorig-
| the model | to retrieve | opposite |     | documents | for the |     |     |     |     |     |     |     |
| --------- | ----------- | -------- | --- | --------- | ------- | --- | --- | --- | --- | --- | --- | --- |
inalPromptrieverdataset.
samequeryunderdifferentinstructions,thissetup • Ins-DV (ours): 240k original instruct sam-
discouragesrelianceonfixedquery-documentas- ples+240kdual-viewsyntheticsamples(size-
| sociations | and | encourages | generalization |     | across |     |     |     |     |     |     |     |
| ---------- | --- | ---------- | -------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
matchedtoIns-orig).
diverseinstructiontypes. • All-orig: 440kinstructsamples+their440k
Duringcontrastivetraining,eachqueryispaired non-instruct counterparts from the original
with its positive document and a set of negatives. Promptriever dataset (each instruct sample
2

was synthesized from a corresponding non- (keyword);andMAIRIFEvalrisesfrom32.14to
| instructsample). |     |     |     |     |     | 36.13. However,FollowIRScoredropsfrom21.33 |     |     |     |     |     |     |
| ---------------- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
• All-DV (ours): 440k original instruct sam- to19.73. SinceIns-DVreplaceshalfoftheoriginal
ples+440kdual-viewsyntheticsamples(size- instructsampleswithDVcounterparts,themodel
matchedtoAll-orig). seesfeweruniquetrainingcontexts,whichmayac-
|          |         |     |         |              |     | count for | this | decline. This | observation |     | points | to  |
| -------- | ------- | --- | ------- | ------------ | --- | --------- | ---- | ------------- | ----------- | --- | ------ | --- |
| Training | Details | We  | use the | Arctic-Embed |     |           |      |               |             |     |        |     |
datadiversityasakeyfactorinsustaininggeneral
framework1
|     | with | 30 hard | negatives | per | query, in- |     |     |     |     |     |     |     |
| --- | ---- | ------- | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
retrievalperformance,ahypothesiswetestdirectly
| cluding | 1–3 instruction | negatives. |     | We  | optimize |     |     |     |     |     |     |     |
| ------- | --------------- | ---------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
intheAll-configurationsbelow.
| with InfoNCE | loss | (van | den Oord | et  | al., 2019) |     |     |     |     |     |     |     |
| ------------ | ---- | ---- | -------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
with temperature τ = 0.02. For both encoders, Mixed-data comparison: All-orig vs. All-syn.
thequeryandinstructionareconcatenatedbefore TheAll-configurationscomparetwostrategiesfor
encoding,whiledocumentsareencodedindepen- scaling the training set to ∼880k while holding
dently. The maximum sequence length is 512 to- theinstructportionfixed(∼440k): supplementing
kensforbothqueriesanddocuments. Allconfig- with non-instruct data (All-orig) versus DV data
urations use the same training protocol to ensure (All-DV).Acrossallbenchmarks,All-DVoutper-
| faircomparisons. |     |     |     |     |     | formsAll-orig. |     | FollowIRp-MRRimprovesfrom |     |     |     |     |
| ---------------- | --- | --- | --- | --- | --- | -------------- | --- | ------------------------- | --- | --- | --- | --- |
5.27to8.30,thehighestamongallconfigurations,
| EvaluationBenchmarks |         |              | Weevaluateonthree |         |         |               |       |            |          |        |     |        |
| -------------------- | ------- | ------------ | ----------------- | ------- | ------- | ------------- | ----- | ---------- | -------- | ------ | --- | ------ |
|                      |         |              |                   |         |         | and Score     | rises | from 20.85 | to       | 21.38. | On  | InfoS- |
| benchmark            | suites: | (1) FollowIR |                   | (Weller | et al., |               |       |            |          |        |     |        |
|                      |         |              |                   |         |         | earch, All-DV |       | achieves   | positive | p-MRR  |     | (31.91 |
2025a),reportingp-MRRforinstructionsensitiv-
length,12.13keyword)whereasAll-origfallsinto
| ity and | an aggregated | Score | for | overall | retrieval |          |           |          |          |     |            |     |
| ------- | ------------- | ----- | --- | ------- | --------- | -------- | --------- | -------- | -------- | --- | ---------- | --- |
|         |               |       |     |         |           | negative | territory | (−23.22, | −49.65), |     | indicating |     |
quality. Thep-MRRmetricquantifiesinstruction
thatthemodelwithnon-instructsupplementation
sensitivitybycomparingamodel’srankingunder
|            |               |     |           |     |          | contradicts | instruction-defined |          |          | relevance. |         | MAIR |
| ---------- | ------------- | --- | --------- | --- | -------- | ----------- | ------------------- | -------- | -------- | ---------- | ------- | ---- |
| two paired | instructions: |     | one where | a   | document |             |                     |          |          |            |         |      |
|            |               |     |           |     |          | metrics     | follow              | the same | pattern: | All-DV     | reaches |      |
is annotated as relevant and one where it is not. 34.08(IFEval)and90.74(InstructIR),boththebest
Positivep-MRRindicatesthemodelcorrectlyad-
acrossallsettings.
justsitsrankinginresponsetoinstructionchanges,
whilenegativevaluesindicateitranksdocuments Theroleofdatadiversity. Theseresults,com-
in the opposite direction. (2) InfoSearch (Zhou binedwiththeIns-comparison,clarifytherespec-
et al., 2025) length and keyword subsets, also re- tive roles of data diversity and instruction super-
portingp-MRR;and(3)MAIR(Sunetal.,2024) vision. TheIns-experimentsshowthatdedicated
IFEval(Zhouetal.,2023)andInstructIR(Ohetal., instructiondatadrivesIFcapability,butreplacing
2024)subsets,reportingnDCG@10. original samples with DV ones reduces diversity
|     |     |     |     |     |     | and costs | general | retrieval | quality. |     | The All- | ex- |
| --- | --- | --- | --- | --- | --- | --------- | ------- | --------- | -------- | --- | -------- | --- |
4 ResultsandAnalysis
|     |     |     |     |     |     | periments | reveal | the converse:      |     | scaling     | with | non-    |
| --- | --- | --- | --- | --- | --- | --------- | ------ | ------------------ | --- | ----------- | ---- | ------- |
|     |     |     |     |     |     | instruct  | data   | provides diversity |     | but dilutes |      | the in- |
Table1summarizesresultsacrossallbenchmarks.
structionsignal,severelydegradingIFwhileoffer-
| We organize | our | analysis | around | two experimen- |     |     |     |     |     |     |     |     |
| ----------- | --- | -------- | ------ | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ingonlymarginalgeneralretrievalbenefitoverIns-
| tal comparisons |     | that together | reveal | the | interplay |     |     |     |     |     |     |     |
| --------------- | --- | ------------- | ------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
orig. All-DVachievesthebestofbothbyprovid-
betweeninstructionsensitivityanddatadiversity.
inginstruction-conditionedtrainingpairsatscale,
Instruct-onlycomparison: IFgainsatthecostof simultaneously maintaining the data volume that
generalretrieval. Inthesize-matchedcompari- sustainsgeneralqualityandtheinstructionsignal
|     |     |     |     |     |     | thatdrivesIFcapability. |     |     | Notably,All-DVcontains |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ---------------------- | --- | --- | --- |
sonbetweenIns-DVandIns-orig(∼480ksamples
each),ourDVdatayieldsconsistentgainsacrossall no non-instruct data, yet achieves the best Score
IFmetrics: FollowIRp-MRRincreasesfrom5.21 across all configurations. This suggests that data
to7.57(+45%),surpassinggeneral-purposemodels volume,ratherthansourceheterogeneityperse,is
suchasEmbeddingGemma-300M(5.61p-MRR) the primary driver of general retrieval quality, as
(Vera et al., 2025) and Qwen3-Embedding-0.6B longasindividualtrainingsamplesaresufficiently
(5.09 p-MRR) (Zhang et al., 2025); InfoSearch diverseintheirquery-documentpairings.
p-MRRimprovesby+122%(length)and+172%
|     |     |     |     |     |     | Cross-backbone |     | generalizability. |     |     | Table | 2 re- |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ----------------- | --- | --- | ----- | ----- |
1https://github.com/snowflakedb/ArcticTraining ports results on bge-m3-retromae, a stronger en-
3

|     |     |     | FollowIR | InfoSearch(p-MRR) |     | MAIR(nDCG@10) |     |     |
| --- | --- | --- | -------- | ----------------- | --- | ------------- | --- | --- |
Trainingdata
|     |              | p-MRR↑ | Score↑ | Length↑ | Keyword↑    | IFEval↑ | InstructIR↑ |     |
| --- | ------------ | ------ | ------ | ------- | ----------- | ------- | ----------- | --- |
|     | Ins-orig     | 5.21   | 21.33  |         | 4.06 2.06   | 32.14   | 89.16       |     |
|     | Ins-DV(ours) | 7.57   | 19.73  |         | 9.02 5.61   | 36.13   | 87.97       |     |
|     | All-orig     | 5.27   | 20.85  | -23.22  | -49.65      | 24.33   | 85.54       |     |
|     | All-DV(ours) | 8.30   | 21.38  |         | 31.91 12.13 | 34.08   | 90.74       |     |
Table 1: Main results on instruction-following retrieval benchmarks. Ins-/All- denote instruct-only and mixed
trainingregimes;-orig/-DVdenoteoriginalandourdual-viewaugmenteddata. Scoreisthemacro-averageacross
threeFollowIRsubsets(MAP@1000ontwosubsets;nDCG@5onone). Negativep-MRRindicatesthemodel
contradictsinstruction-definedrelevance.
|     |     |     | FollowIR | InfoSearch(p-MRR) |     | MAIR(nDCG@10) |     |     |
| --- | --- | --- | -------- | ----------------- | --- | ------------- | --- | --- |
Trainingdata
|     |              | p-MRR↑ | Score↑ | Length↑ | Keyword↑    | IFEval↑ | InstructIR↑ |     |
| --- | ------------ | ------ | ------ | ------- | ----------- | ------- | ----------- | --- |
|     | Ins-orig     | 9.40   | 22.26  |         | 19.00 4.18  | 27.12   | 90.40       |     |
|     | Ins-DV(ours) | 11.47  | 19.76  |         | 28.64 48.42 | 29.37   | 89.53       |     |
|     | All-orig     | 8.84   | 20.69  | -27.08  | -62.04      | 25.68   | 90.67       |     |
|     | All-DV(ours) | 13.92  | 20.99  |         | 40.15 49.62 | 27.07   | 90.80       |     |
Table2: Resultsonbge-m3-retromae. ScoreiscomputedidenticallytoTable1. Thesamepatternshold: DVdata
improvesIF,mixingnon-instructdatadegradesit,andDVaugmentationcounteractsthisdegradation.
coder with a different pretraining strategy. Both thesizing this complement imposes a contrastive
experimental patterns replicate faithfully. In the constraint across instruction space, requiring the
Ins-comparison,Ins-DVimprovesallIFmetrics: queryencodertoresolvewhereI andI di-
|     |     |     |     |     |     |     | orig | new |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- |
FollowIRp-MRRrisesfrom9.40to11.47(+22%), verge,notjustwhichdocumentseachinstruction
InfoSearch keyword p-MRR surges from 4.18 to excludes. This targets instructional distinctions
48.42,whileScoredropsfrom22.26to19.76,con- ratherthaninstructionalexclusions,astructurally
firmingthesameIF/diversitytrade-off. IntheAll- richersupervisorysignal.
| comparison, | the same | degradation | pattern | reap- |            |             |                 |     |
| ----------- | -------- | ----------- | ------- | ----- | ---------- | ----------- | --------------- | --- |
|             |          |             |         |       | A gradient | perspective | on data mixing. | The |
pearswithAll-orig(−27.08and−62.04onInfoS-
datamixingcatastropheoffersamechanisticexpla-
earch),andAll-DVagainreversesitentirely(40.15
nationforwhygeneral-purposeembeddingmodels
and49.62),achievingthebestFollowIRp-MRRof
13.92whilemaintainingcompetitiveScore(20.99). often underperform instruction-specialized ones
|               |                |       |     |           | despitelargerscale. | Non-instructsamplesprovide |                  |     |
| ------------- | -------------- | ----- | --- | --------- | ------------------- | -------------------------- | ---------------- | --- |
| The magnitude | of the keyword | gains | is  | consider- |                     |                            |                  |     |
|               |                |       |     |           | gradient            | signal that rewards        | query-correlated | re- |
ablylargeronbge-m3,suggestingthatastronger
backbone amplifies the benefit of our DV signal. trieval regardless of instructions; at a 50/50 mix,
|                 |             |             |             |      | this overwhelms  | the          | instruction signal.    | Instruc- |
| --------------- | ----------- | ----------- | ----------- | ---- | ---------------- | ------------ | ---------------------- | -------- |
| The cross-model | consistency | confirms    | that        | both |                  |              |                        |          |
|                 |             |             |             |      | tion sensitivity | is therefore | not a capability       | that     |
| the DV method   | and the     | data mixing | degradation |      |                  |              |                        |          |
|                 |             |             |             |      | accumulates      | with scale   | but a fragile property | re-      |
arebackbone-agnosticphenomena,reinforcingthe
generalityofourfindings. quiringconsistent supervision. Thisisconsistent
withInF-IR(Zhuangetal.,2025),whichachieves
| 5 Discussion |     |     |     |     | competitiveIFperformancefrom∼38kspecialized |     |     |     |
| ------------ | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- |
triplets,suggestingsignalpuritymattersmorethan
Polarityreversalversusinstruction-basedneg-
|         |                                     |     |     |     | volume. | OurDVstrategyaddressesthisbyembed- |     |     |
| ------- | ----------------------------------- | --- | --- | --- | ------- | ---------------------------------- | --- | --- |
| atives. | BothPromptriever(Welleretal.,2025b) |     |     |     |         |                                    |     |     |
dinganinstruction-conditioningsignalintoevery
andInF-IR(Zhuangetal.,2025)demonstratethat
trainingpair,providingauniformgradienttoward
instruction-tiednegativesoutperformgenerichard
instruction-conditionedrepresentationseveninthe
| negatives, | buttreatnegativesasfixedfailuresrel- |     |     |     |     |     |     |     |
| ---------- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
presenceofgeneral-retrievaldata.
| ative to a                   | given instruction,                 | i.e., | documents        | that |              |     |     |     |
| ---------------------------- | ---------------------------------- | ----- | ---------------- | ---- | ------------ | --- | --- | --- |
| shouldnotberetrievedunderit. |                                    |       | Polarityreversal |      | 6 Conclusion |     |     |     |
| reframesthis:                | aninstructionnegativeisacondition- |       |                  |      |              |     |     |     |
allyrelevantdocument,onethatshouldberetrieved We present a dual-view data synthesis strategy
underadifferent,complementaryinstruction. Syn- basedonpolarityreversalthatcreatescomplemen-
4

tarytrainingpairsatnoadditionalannotationcost. HongjinSu,WeijiaShi,JungoKasai,YizhongWang,
Experimentsacrosstwoencoderbackbonesyield Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A.
|              |                                   |     |     |     |     |     | Smith,    | Luke Zettlemoyer, |                             | and | Tao Yu. | 2023. One |
| ------------ | --------------------------------- | --- | --- | --- | --- | --- | --------- | ----------------- | --------------------------- | --- | ------- | --------- |
| twoinsights: | (1)dedicatedinstructiondatadrives |     |     |     |     |     |           |                   |                             |     |         |           |
|              |                                   |     |     |     |     |     | embedder, | any               | task: Instruction-finetuned |     |         | text em-  |
IFsensitivity,whiledatadiversitysustainsgeneral
|           |          |     |         |           |            |     | beddings. | InFindingsoftheAssociationforCompu- |     |       |       |            |
| --------- | -------- | --- | ------- | --------- | ---------- | --- | --------- | ----------------------------------- | --- | ----- | ----- | ---------- |
| retrieval | quality, | and | (2) our | synthesis | reconciles |     |           |                                     |     |       |       |            |
|           |          |     |         |           |            |     | tational  | Linguistics:                        | ACL | 2023, | pages | 1102–1121, |
thesecompetingdemands,simultaneouslyimprov- Toronto,Canada.AssociationforComputationalLin-
guistics.
| ing both | dimensions |     | at equal | data | budget. | The |     |     |     |     |     |     |
| -------- | ---------- | --- | -------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
approachrequiresnochangestoexistingpipelines. WeiweiSun,ZhengliangShi,WuJiuLong,Lingyong
|     |     |     |     |     |     |     | Yan, Xinyu           | Ma, | Yiding | Liu,                | Min Cao, | Dawei Yin, |
| --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | ------ | ------------------- | -------- | ---------- |
|     |     |     |     |     |     |     | andZhaochunRen.2024. |     |        | MAIR:Amassivebench- |          |            |
Limitations
|     |     |     |     |     |     |     | markforevaluatinginstructedretrieval. |     |     |     |     | InProceed- |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | ---------- |
The polarity-reversal synthesis assumes that a ingsofthe2024ConferenceonEmpiricalMethodsin
NaturalLanguageProcessing,pages14044–14067,
| meaningful | complementary |     |     | instruction | exists | for |     |     |     |     |     |     |
| ---------- | ------------- | --- | --- | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- |
Miami,Florida,USA.AssociationforComputational
| eachdatapoint;inpractice,ourmanualinspection |     |     |     |     |     |     | Linguistics. |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
foundthistoholdforthevastmajorityofcases,but
AaronvandenOord,YazheLi,andOriolVinyals.2019.
querieswithverynarrowrelevancecriteriamayoc-
Representationlearningwithcontrastivepredictive
casionallyyieldlessnaturalreversals. Weevaluate coding. Preprint,arXiv:1807.03748.
onencoder-basedbi-encoderretrievers;exploring
|     |     |     |     |     |     |     | Henrique | Schechter | Vera, | Sahil | Dua, | Biao Zhang, |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ----- | ----- | ---- | ----------- |
theapplicabilitytodecoder-basedorcross-encoder
|     |     |     |     |     |     |     | Daniel | Salz, Ryan | Mullins, | Sindhu | Raghuram | Pa- |
| --- | --- | --- | --- | --- | --- | --- | ------ | ---------- | -------- | ------ | -------- | --- |
architecturesisanaturaldirectionforfuturework.
nyam,SaraSmoot,IftekharNaim,JoeZou,Feiyang
Additionally, our experiments focus on English- Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas
languagebenchmarks,andextendingtheapproach Gonzalez, Omar Sanseviero, Glenn Cameron, Ian
Ballantyne,KatBlack,KaifengChen,and70others.
tomultilingualsettingsremainsaninterestingav-
|                |     |     |     |     |     |     | 2025. Embeddinggemma: |     |                            | Powerfulandlightweight |     |     |
| -------------- | --- | --- | --- | --- | --- | --- | --------------------- | --- | -------------------------- | ---------------------- | --- | --- |
| enuetoexplore. |     |     |     |     |     |     |                       |     | Preprint,arXiv:2509.20354. |                        |     |     |
textrepresentations.
OrionWeller,BenjaminChang,SeanMacAvaney,Kyle
| References |     |     |     |     |     |     | Lo, Arman                     | Cohan, | Benjamin |     | Van Durme,     | Dawn |
| ---------- | --- | --- | --- | --- | --- | --- | ----------------------------- | ------ | -------- | --- | -------------- | ---- |
|            |     |     |     |     |     |     | Lawrie,andLucaSoldaini.2025a. |        |          |     | FollowIR:Eval- |      |
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun uatingandteachinginformationretrievalmodelsto
Luo, Defu Lian, and Zheng Liu. 2024. M3- followinstructions. InProceedingsofthe2025Con-
ferenceoftheNationsoftheAmericasChapterofthe
| embedding:        |     | Multi-linguality, |            | multi-functionality, |         |       |                                         |     |     |     |     |       |
| ----------------- | --- | ----------------- | ---------- | -------------------- | ------- | ----- | --------------------------------------- | --- | --- | --- | --- | ----- |
|                   |     |                   |            |                      |         |       | AssociationforComputationalLinguistics: |     |     |     |     | Human |
| multi-granularity |     | text              | embeddings |                      | through | self- |                                         |     |     |     |     |       |
knowledge distillation. In Findings of the Asso- Language Technologies (Volume 1: Long Papers),
ciation for Computational Linguistics: ACL 2024, pages11926–11942,Albuquerque,NewMexico.As-
pages2318–2335,Bangkok,Thailand.Association sociationforComputationalLinguistics.
forComputationalLinguistics.
OrionWeller,BenVanDurme,DawnLawrie,Ashwin
|     |     |     |     |     |     |     | Paranjape, | Yuhao | Zhang, | and | Jack Hessel. | 2025b. |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------ | --- | ------------ | ------ |
IvanHabernal,OmniaZayed,andIrynaGurevych.2016.
|           |                                    |     |     |           |               |     | Promptriever: |      | Instruction-trained |         | retrievers       | can be |
| --------- | ---------------------------------- | --- | --- | --------- | ------------- | --- | ------------- | ---- | ------------------- | ------- | ---------------- | ------ |
| C4Corpus: | Multilingualweb-sizecorpuswithfree |     |     |           |               |     |               |      |                     |         |                  |        |
|           |                                    |     |     |           |               |     | prompted      | like | language            | models. | In International |        |
| license.  | In Proceedings                     |     | of  | the Tenth | International |     |               |      |                     |         |                  |        |
ConferenceonLanguageResourcesandEvaluation Conference on Representation Learning, volume
2025,pages17660–17683.
(LREC’16),pages914–922,Portorož,Slovenia.Eu-
ropeanLanguageResourcesAssociation(ELRA).
XinZhang,YanzhaoZhang,DingkunLong,WenXie,
|     |     |     |     |     |     |     | Ziqi Dai, | Jialong | Tang, | Huan | Lin, Baosong | Yang, |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | ----- | ---- | ------------ | ----- |
HanseokOh,HyunjiLee,SeonghyeonYe,HaebinShin,
|        |       |           |     |          |         |      | Pengjun              | Xie, | Fei Huang, | Meishan               | Zhang, | Wenjie |
| ------ | ----- | --------- | --- | -------- | ------- | ---- | -------------------- | ---- | ---------- | --------------------- | ------ | ------ |
| Hansol | Jang, | Changwook |     | Jun, and | Minjoon | Seo. |                      |      |            |                       |        |        |
|        |       |           |     |          |         |      | Li,andMinZhang.2024. |      |            | mGTE:Generalizedlong- |        |        |
2024. Instructir: A benchmark for instruction fol- contexttextrepresentationandrerankingmodelsfor
lowing of information retrieval models. Preprint, InProceedingsofthe2024
multilingualtextretrieval.
arXiv:2402.14334.
|                |     |                       |     |     |     |           | Conference       | on  | Empirical                     | Methods | in  | Natural Lan- |
| -------------- | --- | --------------------- | --- | --- | --- | --------- | ---------------- | --- | ----------------------------- | ------- | --- | ------------ |
|                |     |                       |     |     |     |           | guageProcessing: |     | IndustryTrack,pages1393–1412, |         |     |              |
| QwenTeam.2025. |     | Qwen3technicalreport. |     |     |     | Preprint, |                  |     |                               |         |     |              |
Miami,Florida,US.AssociationforComputational
| arXiv:2505.09388. |     |     |     |     |     |     | Linguistics. |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
Colin Raffel, Noam Shazeer, Adam Roberts, Kather- YanzhaoZhang,MingxinLi,DingkunLong,XinZhang,
ine Lee, Sharan Narang, Michael Matena, Yanqi Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Zhou,WeiLi,andPeterJ.Liu.2020. Exploringthe DayihengLiu,JunyangLin,FeiHuang,andJingren
limitsoftransferlearningwithaunifiedtext-to-text Zhou. 2025. Qwen3 embedding: Advancing text
transformer. JournalofMachineLearningResearch, embeddingandrerankingthroughfoundationmodels.
| 21(140):1–67. |     |     |     |     |     |     | Preprint,arXiv:2506.05176. |     |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- |
5

JeffreyZhou,TianjianLu,SwaroopMishra,Siddhartha
inaspecificlanguage),explicitexclusions(e.g.,“excludemajor
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and capitals”,“excludebeach/islandtopics”).
LeHou.2023. Instruction-followingevaluationfor 3) Diversityrequirement
Ensurethenewinstruction’sformatandperspectivedifferfrom
largelanguagemodels. Preprint,arXiv:2311.07911. theoriginal_instruction(e.g.,switchvoice,deliverabletype,
constraintstyle).Avoidtrivialrewording.
Jianqun Zhou, Yuanlei Zheng, Wei Chen, Qianqian 4) Sanitychecks
Zheng,ShangZeyuan,WeiZhang,RuiMeng,and • WouldP+befilteredoutbythenewconstraints?Ifnot,tighten
them.
XiaoyuShen.2025. Beyondcontentrelevance: Eval- • IsN⋆clearlyincludedbythepositiveselectors? Ifnot,pivot
uatinginstructionfollowinginretrievalmodels. In constraintstowardN⋆’sattributes.
InternationalConferenceonRepresentationLearn-
• DoallNistillgetexcluded?Ifanyslipin,addexplicitexclusions
ortightenscope.
ing,volume2025,pages84965–84996. 5) Conciseness
Thenewinstructionmustbeoneortwosentences,imperative,
Yuchen Zhuang, Aaron Trinh, Rushi Qiang, Haotian concrete,andunambiguous.
Sun, Chao Zhang, Hanjun Dai, and Bo Dai. 2025. Guardrails
Towardsbetterinstructionfollowingretrievalmodels. • DonotreferencepassageIDsorthismeta-task;excludeorincludeby
attributeonly.
Preprint,arXiv:2505.21439.
• Donotmodifythequeryorpassages.
A ThePromptTemplateforData Checklist(beforeyououtput)
• P+becomesaninstructionnegative.
Synthesis • N⋆becomespositive.
• AllNiremaininstructionnegatives.
Thefollowingboxshowsthefullprompttemplate • Instruction is concise (≤2 sentences) and diverse vs. the
original_instruction.
used for polarity-reversed instruction synthesis. • Nometa-tasklanguageorpassageIDsappearintheinstruction.
Templatevariables(inmonospace)arepopulated Nowperformthetaskwiththefollowingdata:
• Query:{{ query }}
perinstance.
• Originalinstruction:{{ original_instruction }}
• Originalpositivepassage:{{ original_positive }}
• Specific instruction negative passage: {{
Goal specific_instruction_negative }}
Createanewsyntheticinstructionthatreversestheoriginalrelevance • Allremaininginstructionnegativepassages:
judgmentonlyforthespecifiedpassages: {% for negative in remaining_negatives %}
• Theoriginalpositivepassage(P+)mustbecomeaninstruction Negativepassage{{ loop.index }}:{{ negative }}
negativeunderthenewinstruction(i.e.,relevanttothepurequerybut {% endfor %}
irrelevantoncetheinstructionisapplied).
• Thespecificinstructionnegative(N⋆)mustbecomethenewpositive
(i.e.,relevanttothepurequeryandtothequery+instruction).
• Allremaininginstructionnegatives(N1...Nk)mustremainin-
structionnegatives.
Youmustnotchangethequeryoranypassagecontent.Onlywriteanew
instruction.
Inputs
• query(string)
• original_instruction(string)
• positive_passage=P+(string)
• specific_instruction_negative=N⋆(string)
• remaining_instruction_negatives=N1...Nk (array; maybe
empty)
Output
Youshouldreasonstepbystep,andthefinalanswershouldbeinthe
followingXMLformat:
<answer>
<new_instruction>[yournewinstruction]</new_instruction>
</answer>
Ifyouthinkthistaskistoohardtoachieve,youshouldsimplyreturn
<answer>None</answer>.
Definitions
• Relevanttothepurequery:reasonablysatisfiestheuser’sintentwith-
outanyextrainstruction.
• Instructionnegative:relevanttothepurequery,butexcludedbythe
instruction(e.g.,byscope,geography,timeframe,format,source
constraints,audiencelevel).
Method
1) Profilepassages
• IdentifyattributesofP+(domain,geography,timeframe,audience,
medium/format,methodology,sources,constraints).
• IdentifyattributesofN⋆thatdistinguishitfromP+.
• SkimeachNitonoteattributesyoumustkeepexcluded.
2) Choosereversallevers
Craftanewinstructionthat:
• PositivelyselectsN⋆’sattributes(soN⋆becomespositive),and
• ExcludesP+via≥1hard,objectiveconstraint(soP+becomes
aninstructionnegative),while
• DoesnotinadvertentlyadmitanyNi(keeptheminstruction
negatives).
Usefullevers:domainnarrowing,region,timeframe/recency,audi-
encelevel,style/format(e.g.,“equationsonly”,“bulletedchecklist”),
methodology/evidencetype,requiredartifacts(e.g.,runnablecode
6
