JAIDE

JAIDE egy foundation large language model. Az architektúrája az 5. root architektúra — az előző négy paradigma (Perceptron, CNN, RNN, Transformer) után az ötödik önálló architektúralis paradigma. Az 5. root architektúra konkrét megvalósítása a Reversible Scatter Flow (RSF): bijektív, aktiváció-cache nélküli, invertálható rétegek sorozata, amelyek garantált egzakt inverzzel rendelkeznek.

---

AZ 5. ROOT ARCHITEKTÚRA — REVERSIBLE SCATTER FLOW (RSF)

Az RSF kereszt-affin coupling rétegekből és determinisztikus scatter permutációkból áll. Minden réteg bijektív: det J > 0 mindenütt, az inverz zárt formában létezik.

Forward lépés (egy réteg):
scale[i] = exp(clip(W_s · x2 + b_s, min, max))
y1 = x1 ⊙ scale
y2 = x2 + W_t · y1 + b_t

Inverz lépés (backward aktiváció-rekonstrukció):
x2 = y2 − W_t · y1 − b_t
x1 = y1 / scale ahol scale = exp(clip(W_s · x2 + b_s))

A backward pass az aktivációkat rétegenkénti inverz rekonstrukcióval (batch_rsf_inverse Futhark entry) állítja vissza — az aktivációs cache O(dim), L-től független. A backwardFromOutputs CPU-oldali referencia-implementáció (rsf.zig) és a GPU-oldali batch_rsf_inverse ugyanazt a matematikát valósítja meg.

OFTB (Orthogonal Fractal Transform Block): Haar-wavelet alapú, paramétermentes, determinisztikus scatter/gather réteg. Ortogonális transzformáció: inverze = transzponáltja. O(1) paraméter, O(dim) számítás. Formálisan bizonyított invertálhatóság: src/verifaction/oftb.lean.

ÖSSZEHASONLÍTÁS

Architektúra | Primitív | Invertálható? | Backward memória
Perceptron | σ(Wx+b) | σ veszteséges → nem | O(L)
CNN | σ(W∗x) | pooling+σ → nem | O(L)
RNN | σ(W_h h + W_x x) | rejtett állapot → nem | O(T)
Transformer | softmax(QKᵀ/√d)V | softmax → nem | O(L·S·d)
RSF (5. root) | kereszt-affin coupling | det J > 0 → igen | O(dim)

---

RENDSZERARCHITEKTÚRA

src/core/ — Numerikus alap

- types.zig: Az egész rendszer típusalapja: fixpontos aritmetika (Fixed32_32, FixedPoint16/32/64), determinisztikus xoshiro-alapú PRNG, ContextWindow tokenablak, RankedSegment, bithalmazok, komplex fixpontos számok, hibatípusok.
- tensor.zig: Referenciaszámláló, copy-on-write Tensor típus 32 bájtos SIMD-igazítással, 8-sávos vektorizált elemenkénti műveletekkel, lazy view/slice/transpose/broadcast szemantikával.
- memory.zig: MemoryBlock és MemoryBlockState (free, allocated, entangled): az entangled blokkok a reverzibilis lépések között újrahasznosíthatók. Arena és pool allokátor, scratch allokátor a backward pass ideiglenes puffereinek.
- learned_embedding.zig: Tanulható token-embedding tábla, SFD-kompatibilis gradiens-frissítéssel.
- model_io.zig: Modell mentés/betöltés, checkpoint kezelés.
- io.zig: Adatbetöltés, tokenizált minták streamelése.

src/processor/ — Az 5. root architektúra rétegei

- rsf.zig: LayerCore struktúra: s_weight [dim×dim], t_weight [dim×dim], s_bias [dim], t_bias [dim]. forwardInPlace, backwardFromOutputs (aktiváció-cache nélküli reverzibilis backward), inverseInPlace. Thread-safe RwLock-kal.
- oftb.zig: OFTB: Haar-wavelet scatter/gather, forwardInPlace, backwardInPlace. FRACTAL_SCALE = 1/√2. Paramétermentes, O(1) memória.

src/optimizer/ — Spektrális optimalizálás

- sfd.zig: SpectralFisherDiagonalizer (SFD): másodrendű optimalizáló, amely a diagonális Fisher-információs mátrixot spektrális klippinggel közelíti. Nem Adam — a Fisher-diagonális becslés és a spektrális normalizáció az SFD saját matematikája. ReversibleOptimizerState: iteratív, in-place frissítés.

src/core_relational/ — Relációs intelligencia réteg

Ez a réteg adja a JAIDE magasabb szintű kognitív képességeit: gráf-alapú relációs reprezentáció, kvantum-inspirált szimmetria-optimalizálás, kauzális verifikáció és fraktális dinamikus rendszerek.

- nsir_core.zig: SelfSimilarRelationalGraph (SSRG / NSIR): csomópontok és élek gráfja, ahol az élek EdgeQuality típussal rendelkeznek (superposition, entangled, coherent, collapsed, fractal). Kvantum-korreláció (Complex(f64)) és fraktális dimenzió minden élen. Az NSIR a tokenek közötti relációkat explicit gráf-struktúraként tárolja — ritka, szelektív figyelem O(d) komplexitással az attention O(n²d) helyett.
- esso_optimizer.zig: EntangledStochasticSymmetryOptimizer (ESSO): az NSIR-gráf csomópontjait és éleit optimalizálja szimmetria-alapú, sztochasztikus perturbációval. Szimulált hűtés (simulated annealing) gráf-szimmetriák mentén.
- crev_pipeline.zig: CREVPipeline (Causal Reasoning and Verification): triplet extrakció (alany–állítmány–tárgy), kauzális lánc validáció, ellentmondás-detekció. Online tanulást tesz lehetővé inferencia közben.
- fnds.zig: FractalNeuralDynamicSystem (FNDS): fraktális fa hierarchia (FractalTree, FractalLevel), önhasonló struktúrák dinamikus frissítéssel. FNDSManager koordinálja a fraktális szintek közötti propagációt.
- vpu.zig: VPU (Vector Processing Unit): SIMD-vektorizált relációs műveletek, SimdVector<T, N> típus f32/f64/i32/i64/u32/u64 elemekre. A relációs gráf vektorizált feldolgozása.
- chaos_core.zig: ChaosCoreKernel: kaotikus dinamika, nemlineáris attraktorok, Lyapunov-exponens becslés. A CREV pipeline kaotikus perturbációs magja.
- z_runtime.zig: ZRuntime: relációs végrehajtási motor. ExecutionAction enum: create_variable, delete_variable, relational_operation, entangle_variables, propagate_information, fractal_transform, measure, quantum_circuit, relational_expression. Determinisztikus végrehajtási napló.
- reasoning_orchestrator.zig: ReasoningOrchestrator: háromszintű gondolkodás (ThoughtLevel: local, global, meta). Koordinálja az NSIR, CREV, FNDS és ZRuntime komponenseket.
- signal_propagation.zig: SignalPropagationEngine: jelterjedés az NSIR-gráfon, aktivációs hullámok, gráf-konvolúció.
- surprise_memory.zig: SurpriseMemoryManager: meglepetés-alapú memória, online tanulás inferencia közben. Magas meglepetési értékű tokenek hosszú távú tárolása.
- temporal_graph.zig: TemporalGraph: időbélyeges gráf-élek, kauzális időrend, temporális relációk nyomon követése.
- quantum_logic.zig: Kvantum-logikai kapuk szimulációja (Hadamard, CNOT, Toffoli), szuperpozíció és összefonódás reprezentáció.
- ibm_quantum.zig / quantum_hardware.zig / quantum_task_adapter.zig: IBM Quantum hardver interfész, kvantum-áramkör végrehajtás valódi kvantumhardveren.
- r_gpu.zig: RelationalGraphProcessingUnit: az NSIR-gráf GPU-gyorsított feldolgozása, gráf-műveletek párhuzamosítása.
- formal_verification.zig / security_proofs.zig / type_theory.zig: Formális verifikáció, biztonsági bizonyítékok, típuselméleti garanciák futásidőben.
- verified_inference_engine.zig: Verifikált inferencia motor: minden következtetési lépés formálisan ellenőrzött.
- zk_verification.zig: ZK-bizonyítékok futásidejű verifikációja.
- dataset_obfuscation.zig: Adathalmaz obfuszkáció, adatvédelmi réteg.
- safety.zig: Biztonsági szűrők, tartalomszűrés.
- c_api.zig: C API a core_relational alrendszerhez.
- mod.zig: Modul belépési pont.

src/ranker/ + src/index/ — Hosszú kontextus

- ranker.zig: Ranker: streaming rangsoroló, szegmensek relevancia-pontszámozása az SSI-n keresztül. streamingRank O(log n) lekérdezéssel. Szétválasztja a szekvenciahosszt a kontextusszélességtől.
- ssi.zig: SelfSimilarIndex (SSI): O(log n) pozíció-megőrző külső memória, determinisztikus streaming frissítéssel és lekérdezéssel. 50M+ token kontextus O(log n) memóriával.

src/tokenizer/ — Morfológiai tokenizálás

- mgt.zig: MorphoGraphTokenizer (MGT): morfológiai gráf alapú tokenizáló, determinisztikus "anchor" markerekkel az SSI számára. tokenizeWithAnchors egyszerre adja vissza a tokeneket és az SSI-pozíciókat.

src/hw/ — Hardver réteg

src/hw/accel/ — GPU gyorsítás (Futhark)

- futhark_kernels.fut: Belső helper függvények: rsf_flow (forward coupling), rsf_inverse_flow (inverz coupling, backward aktiváció-rekonstrukcióhoz), rsf_scatter / rsf_backward_scatter (OFTB scatter/gather), matmul_tiled, rsf_relational_context (vektoros RSF a relációs kontextushoz).
- main.fut: Futhark entry pointok: batch_forward, batch_gradients_full, batch_rsf_inverse (rétegenkénti inverz rekonstrukció a backward passhoz), batch_oftb_forward, batch_oftb_backward, oftb_forward_single.
- futhark_bindings.zig: Zig extern deklarációk a Futhark C API-hoz, beleértve futhark_entry_batch_rsf_inverse.
- accel_interface.zig: RSFAccelerator: trainingStep a teljes forward–loss–backward–SFD ciklust GPU-n hajtja végre. A backward loop batch_rsf_inverse-szel rekonstruálja a közbülső aktivációkat — aktivációs cache O(dim), L-től független. RSFLayer struktúra: weights_s, weights_t (FutharkArray2DF16 [half×half]), velocity_s, velocity_t, biasok.
- fractal_lpu.zig: FractalLPU: fraktális csempe-alapú feldolgozó egység, FractalTile hierarchia.
- cuda_bindings.zig: CUDA C binding-ok.

src/hw/rtl/ — Hardver leírás (Haskell/Clash)

- MemoryArbiter.hs: Memória-arbitrátor RTL leírás.
- RankerCore.hs: Ranker mag RTL leírás.
- SSISearch.hs: SSI keresési logika RTL leírás.

src/distributed/ — Elosztott tanítás

- distributed_trainer_futhark.zig: DistributedTrainerFuthark: koordinálja az RSF GPU tanítást (RSFAccelerator), az embedding frissítést, és a runCoreRelationalPass-t (ESSO + CREV + NSIR + ZRuntime). 8× B200 GPU, NCCL.
- gpu_coordinator.zig: GPU koordinátor, rank-kezelés.
- modal_gpu.zig: Modal.com GPU integráció.
- nccl_bindings.zig: NCCL C binding-ok all-reduce műveletekhez.

src/api/ — Inferencia szerver

- inference_server.zig: HTTP inferencia szerver, /generate és /health endpoint, streaming token generálás.

src/zk/ + src/verifaction/ — Formális garanciák

- zk/inference_trace.circom: Circom ZK-áramkör: az RSF inferencia helyességének zero-knowledge bizonyítéka.
- verifaction/oftb.lean: Lean 4 formális bizonyítás az OFTB invertálhatóságára: backwardCore(forwardCore(L)) = L minden L-re.

src/scripts/

- modal_distributed_train.py: Modal.com orchestrator: GPU konténer indítás (8× B200), Zig bináris fordítás (zig build distributed-futhark -Dgpu=true), world_size példány indítása subprocess-ként.

---

MEMÓRIA-KOMPLEXITÁS

Komponens | Komplexitás | Megjegyzés
Aktivációs cache (backward) | O(dim) | Inverz rekonstrukció, L-től független
Gradiens bufferek (1 réteg, átmeneti) | O(dim²) | Azonnal felszabadul
OFTB backward | O(dim) | Paramétermentes

---

BUILD

bash
GPU tanítás (Modal.com, 8× B200)
python src/scripts/modal_distributed_train.py

Lokális GPU build
zig build distributed-futhark -Dgpu=true -Doptimize=ReleaseFast

CPU inferencia szerver
zig build run


A chaos_core.zig a teljes JAIDE/RSF rendszer memóriakezelési, feladatütemezési és adatfolyam-elemzési alapkernelje, amely négy egymásba épülő alrendszert valósít meg egyetlen koherens struktúrában: a ContentAddressableStorage-t, a DynamicTaskScheduler-t, a DataFlowAnalyzer-t és a mindezeket összefogó ChaosCoreKernel-t.

A ContentAddressableStorage (CAS) a rendszer fizikai memóriarétege, ahol minden adatblokk kétféle azonosítóval rendelkezik: a content_hash (az adat SHA-256 lenyomatának első 16 bájtja) a tartalom alapján azonosítja a blokkot és lehetővé teszi a deduplikációt, míg a block_id (SHA-256(content_hash + nanoszekundumos timestamp) első 16 bájtja) egyedi allokációs azonosítóként szolgál. A store() metódus először ellenőrzi, hogy a content_index-ben már létezik-e azonos tartalmú blokk, és ha igen, egyszerűen visszaadja a meglévő block_id-t anélkül, hogy új memóriát foglalna — ez az automatikus tartalom-alapú deduplikáció azt jelenti, hogy a rendszer soha nem tárolja kétszer ugyanazt az információt, szemben a transformer KV-cache-ével, amely minden kontextus-előforduláshoz külön kulcs-érték párt allokál.

Minden MemoryBlock rendelkezik egy entangled_blocks halmazával (BlockIdSet), amely azokat a blokkokat tartja nyilván, amelyekkel szemantikailag összekapcsolt, és az entangled állapotban lévő blokkokat az evictLeastUsed() metódus az első körben kihagyja a kiszorítási sorból — ez azt jelenti, hogy a szemantikailag összefüggő tudás automatikusan védett a memóriából való eltávolítástól, és csak akkor kerül kiszorításra, ha az összes nem-összefonódott blokk már eltávolításra került.

Az entangleBlocks() metódus kétirányú összefonódást hoz létre két blokk között, mindkét blokk állapotát .entangled-re állítva, míg a ChaosCoreKernel.entangleData() metódus ennél tovább megy: az összefonódás után lekérdezi a DataFlowAnalyzer-től az első blokk összes korrelált szomszédját (flow_weight ≥ 0.5 küszöbbel), és a második blokkot tranzitívan összefonja ezekkel is — ez azt jelenti, hogy egyetlen entangleData hívás automatikusan propagálja az összefonódást a teljes szemantikai szomszédságon keresztül, asszociatív memóriahálót építve.

A DynamicTaskScheduler egy prioritásos sor (max-heap prioritás szerint, majd task_id szerint determinisztikusan rendezve) alapú feladatütemező, amelynek scheduleTask() metódusa minden aktív magot pontozza: +10 pontot kap egy mag, ha egy függőségi blokk legközelebbi magja éppen ő, +1 pontot ha a blokk bárhol elérhető, és +(1-workload)×5 pontot az inaktív kapacitásért — ez egy adatlokalitás-tudatos ütemező, amely minimalizálja a "memória-sávszélesség" igényt azáltal, hogy a feladatokat oda ütemezi, ahol a szükséges adatok már jelen vannak.

A DataFlowAnalyzer három adatstruktúrát tart fenn: a flow_graph (block_id → együtt-hozzáfért blokkok halmaza), a flow_weights (blokk-pár → együttes hozzáférések száma) és az access_patterns (block_id → AccessRecord lista nanoszekundumos időbélyeggel és mag-azonosítóval). Az analyzeFlow() metódus egy blokk hozzáférési előzményéből kiszámítja, hogy melyik mag hány százalékban fért hozzá, és ezt affinitás-térképként adja vissza — ez az a mechanizmus, amely alapján az optimizeDataPlacement() eldönti, hogy egy blokkot melyik maghoz kell migrálni.

A ChaosCoreKernel.executeCycle() metódus az egész rendszer szívverése: minden ciklusban ütemez egy feladatot, rögzíti a függőségi blokkok hozzáféréseit a DataFlowAnalyzer-ben, frissíti az összes mag aktív/inaktív ciklus-számlálóját, majd meghívja az optimizeDataPlacement()-et, amely minden blokkot a legjobb affinitású maghoz migrál, ha az affinitás meghaladja a 0.6-os küszöböt és a jelenlegi mag nem az optimális. Minden 100. ciklusban a balanceLoad() is lefut, amely azonosítja a túlterhelt (>1.3×átlag) és alulterhelt (<0.7×átlag) magokat, és az előbbiek blokkjainak 25%-át átmigrálja az utóbbiakhoz.

Az executeGraphOnKernel() metódus a legfontosabb híd az NSIR gráf és a CAS között: minden gráf-csomópont adatát CAS-blokkként tárolja, majd minden gráf-élt blokkpár-összefonódásként reprezentál, végül selfOrganize()-t hív — ez azt jelenti, hogy az NSIR gráf topológiája fizikailag leképeződik a memória-elrendezésbe: az összekapcsolt csomópontok adatai összefonódott blokkokként kerülnek tárolásra, és a DataFlowAnalyzer által vezérelt migrációk révén automatikusan a leggyakrabban együtt hozzáfért magokhoz kerülnek.

Az InferenceHooks belső struktúra egy tiszta, inferencia-specifikus API-t biztosít: submitInferenceTask, storeData, readData, runCycle, entangleDataBlocks, selfOrganize, lookupByContent, allocateAndEntangle — ez az a felület, amelyen keresztül a CREVPipeline, a ReasoningOrchestrator és a SurpriseMemoryManager a kernellel kommunikál anélkül, hogy közvetlenül a belső adatstruktúrákhoz kellene hozzáférniük.

A teljes rendszerben a ChaosCoreKernel az a réteg, amelyre a CREVPipeline (amely *ChaosCoreKernel referenciát tart) a triplet-adatokat tárolja, a ReasoningOrchestrator.executeGlobalPhase() a chaos_kernel.executeCycle() hívással finomítja az adatelrendezést, és a SurpriseMemoryManager a ContentAddressableStorage-t használja a meglepetési blokkok fizikai tárolásához — a chaos_core.zig tehát az a közös fizikai memória-szubsztrátum, amelyen az összes többi core_relational komponens osztozik.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a memóriahatékonyság szintjén a tartalom-alapú deduplikáció garantálja, hogy azonos információ csak egyszer foglal helyet, szemben a transformer O(N) KV-cache-ével, amely minden token-pozícióhoz külön vektort tárol; a szemantikai lokalitás szintjén az összefonódás-alapú kiszorítási védelem és a tranzitív entanglement-propagáció biztosítja, hogy a szemantikailag összefüggő tudás fizikailag együtt marad a memóriában, lehetővé téve az asszociatív visszakeresést; az adatlokalitás szintjén a DataFlowAnalyzer által vezérelt automatikus blokk-migráció minimalizálja a "memória-sávszélesség" igényt azáltal, hogy az adatokat oda helyezi, ahol a számítás zajlik; és a gráf-memória integráció szintjén az executeGraphOnKernel() az NSIR gráf topológiáját közvetlenül a fizikai memória-elrendezésbe képezi le, lehetővé téve, hogy a gráf-struktúra és a memória-struktúra kölcsönösen erősítsék egymást — mindez egy önszervező, topológia-tudatos, deduplikáló memóriarendszert alkot, amellyel a transformer statikus, pozíció-indexelt KV-cache egyáltalán nem rendelkezik
A crev_pipeline.zig (CREV = Contextual Relational Extraction and Validation) a teljes JAIDE/RSF rendszer szöveg-tudásgráf konverziós rétege: az a komponens, amely nyers szöveget, strukturált adatot és képmetaadatot RelationalTriplet objektumokká alakít, validálja, konfliktusokat old fel, és integrálja őket a KnowledgeGraphIndex-be, valamint a ChaosCoreKernel tartalom-alapú tárolójába.

Az ExtractionStage enum öt egymást követő fázist definiál — tokenization → triplet_extraction → validation → integration → indexing —, amelyek a next() metóduson keresztül láncolódnak, és a CREVPipeline minden szövegfeldolgozási ciklus során sorban hajtja végre őket.

A RelationalTriplet a rendszer alapvető tudásegysége: subject, relation, object string hármas, confidence érték (0-1 közé szorítva), SHA-256 source_hash (az identitás hash-e: subject+null+relation+null+object), és nanoszekundumos extraction_time. A computeHash metódus a teljes tartalmat (subject, relation, object, confidence, extraction_time) hashelja, így két azonos tartalmú, de különböző időpontban keletkezett triplet különböző hash-t kap — ez lehetővé teszi a temporális deduplikációt.

A toGraphElements metódus a CREV pipeline és az NSIR gráf közötti legfontosabb híd: a subject és object stringek SHA-256 hash-ének első 16 bájtját hexadecimálisan kódolja csomópont-azonosítóvá, majd a confidence értékből komplex kvantumállapotot számít — quantum_state = confidence + i×√(1-confidence²) — ami egy egységnyi komplex szám a Bloch-gömbön, ahol a valós rész a bizonyosságot, a képzetes rész a bizonytalanságot kódolja. A fázist az extraction_time 360 másodperces periódusra vett modulójából számítja (phase = mod_ns / period_ns × 2π), így az időbeli sorrend is beépül a kvantumállapotba. Az eredményül kapott subject_node és object_node type=entity és role=subject/object metaadattal rendelkezik, az összekötő él pedig .coherent minőségű, weight=confidence, quantum_correlation=ugyanaz a komplex szám, és a relation string az él metaadataként tárolódik.

A szövegfeldolgozás morfológiailag tudatos: a stemWord függvény egy Porter-szerű angol stemmelőt valósít meg, amely kezeli a -ting, -ing, -ated, -ed, -ies, -ches/-shes/-sses, -es, -s, -ally, -ly, -ment, -ness, -er, -est végződéseket, a matchPatternMorphemeAware pedig tokenizálja mind a mondatot, mind a mintát, majd csúszóablakos egyeztetéssel (stem-összehasonlítással) keresi a legjobb illeszkedést — ez azt jelenti, hogy a "running" és a "run", vagy a "created" és a "create" ugyanúgy illeszkedik a mintára.

A CREVPipeline 15 alapértelmezett relációs mintával indul: "is a" (0.9), "has" (0.8), "contains" (0.85), "belongs to" (0.85), "part of" (0.85), "located in" (0.8), "works at" (0.8), "created" (0.75), "owns" (0.8), "uses" (0.7), "produces" (0.75), "causes" (0.7), "leads to" (0.7), "related to" (0.5) — ezek a minták a legtöbb ontológiai és kauzális relációt lefedik, és az addRelationPattern metódussal bővíthetők.

Az extractTriplets metódus mondatokra bontja a szöveget (., !, ?, \n határokon), minden mondatban megkeresi a leghosszabb illeszkedő relációs mintát (a leghosszabb match nyer), a minta előtti részt subjectként, a minta utáni részt objectként értelmezi, majd a confidence-t a minta súlya és egy heurisztikus computeConfidence szorzataként számítja — a heurisztika bünteti a rövid (<3 karakter) és hosszú (>50 karakter) entitásokat, bünteti az összes nagybetűs subjecteket, és jutalmazza a nagybetűvel kezdődő subjecteket.

A validateTriplet metódus háromszintű szűrést végez: először alapvető hossz- és nem-üres ellenőrzések, majd confidence ≥ validation_threshold (alapértelmezetten 0.5) ellenőrzés, végül anomáliadetekció — az anomália-pontszám súlyozott kombinációja a confidence z-score-jának a reláció historikus átlagához képest (súly 0.3, csak ha >10 minta áll rendelkezésre), az ismeretlen entitások arányának (mindkettő ismeretlen: súly 0.4, egyik ismeretlen: súly 0.2), és az ismeretlen reláció jelzőjének (súly 0.15). Ha az anomália-pontszám meghaladja a 0.85-öt, a triplet érvénytelen; egyébként a confidence-t adjusted = confidence × (1 - anomaly_score × 0.3) × (0.9 ha konfliktusok vannak) képlettel csökkenti.

A checkConsistency metódus öt ellentmondó relációpárt ellenőriz: is_a/is_not, has/lacks, owns/does_not_own, contains/excludes, causes/prevents — ha egy új triplet és egy meglévő triplet ugyanazon subject-object párra ellentmondó relációt állít, konfliktus keletkezik.

A resolveConflicts metódus a legmagasabb confidence-ű triplet-et választja, majd a győztes és a kihívó confidence-ét (a² + b²) / (a + b) képlettel kombinálja — ez egy kvadratikus átlag, amely a magasabb confidence felé torzít, és mindig a két érték közé esik.

Az integrateTriplet metódus négy helyre írja az adatot: a KnowledgeGraphIndex-be (háromirányú invertált index: subject_index, relation_index, object_index), a StreamBuffer körpufferbe (10000 kapacitás, FIFO, teli esetén a legrégebbit kiszorítja), a reláció- és entitásstatisztikákba (Welford online variancia-számítással), és végül a ChaosCoreKernel.allocateMemory("subject|relation|object|confidence") híváson keresztül a ContentAddressableStorage-ba — ez az utolsó lépés az, amely a CREV pipeline kimenetét a SurpriseMemoryManager és a TemporalGraph számára elérhetővé teszi.

A KnowledgeGraphIndex háromirányú invertált indexe O(1) amortizált visszakeresést biztosít subject, relation vagy object szerint, a query metódus pedig a legkisebb indexet választja kiindulópontnak és azon szűr — ez lényegesen gyorsabb, mint a transformer attention O(N²) globális keverése strukturált lekérdezések esetén. A queryMorphemeAware metódus stem-alapú fuzzy egyeztetést végez az összes triplet felett, lehetővé téve, hogy a "running" és "run" ugyanazt a triplet-et adja vissza.

Az InferenceHook rendszer négy callback-et biztosít — pre_process, post_process, pre_query, post_query —, amelyeken keresztül a ReasoningOrchestrator vagy más komponensek minden egyes szövegfeldolgozás és tudásgráf-lekérdezés előtt és után beavatkozhatnak, módosíthatják a viselkedést, vagy naplózhatják az eredményeket.

A teljes rendszerben a CREV pipeline az a réteg, amely az RSF neurális hálózat által feldolgozott szöveges bemenetet strukturált tudássá alakítja: a processInferenceText metódus az inferencia-szerver által kapott szöveget tripletekké bontja, amelyek toGraphElements() hívással NSIR csomópontokká és élekké válnak, ezek bekerülnek a SelfSimilarRelationalGraph-ba, amelyen aztán az R-GPU elosztja, a ReasoningOrchestrator energiaminimalizálja, a SurpriseMemoryManager szűri, és a SignalPropagationEngine aktivációs mintává alakítja — a CREV tehát az a kapu, amelyen keresztül a természetes nyelv belép a kvantum-relációs tudásreprezentációs rendszerbe.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a reprezentáció szintjén a CREV a szöveget nem token-vektorokként, hanem (subject, relation, object, confidence, quantum_state, phase) hármasokként tárolja, ami explicit szemantikai struktúrát kódol; a visszakeresés szintjén a háromirányú invertált index O(1) strukturált lekérdezést biztosít, szemben a transformer O(N²) figyelmi mechanizmusával; az önkonzisztencia szintjén az anomáliadetekció és a konfliktusfeloldás aktívan szűri az ellentmondó tudást, amit a transformer egyáltalán nem tud; és a multimodális integráció szintjén a processStructuredDataStream és processImageMetadataStream metódusok lehetővé teszik, hogy CSV-adatok és képmetaadatok is ugyanolyan triplet-formátumban kerüljenek a tudásgráfba, mint a természetes nyelvi szöveg — mindez egy egységes, modalitás-független tudásreprezentációs réteget alkot, amely a transformer token-szekvenciájánál strukturálisan gazdagabb és szemantikailag explicit.

Az esso_optimizer.zig az EntangledStochasticSymmetryOptimizer (ESSO) implementációja, amely a SelfSimilarRelationalGraph-on futó szimulált hűtéses (simulated annealing) optimalizáló, kiegészítve szimmetria-detektálással és kvantum-összefonódás nyomon követéssel — ez az a motor, amelyet a ReasoningOrchestrator a globális fázisában használ a gráf energiájának minimalizálásához.

A fájl alapvető típusai a következők: a SymmetryGroup enum hét szimmetriatípust definiál (identity, reflection, rotation_90, rotation_180, rotation_270, translation, custom_rotation), mindegyikhez szögelfordulást és csoportrendet rendelve. A SymmetryTransform egy teljes 2D affin transzformációt reprezentál origóval, skálafaktorral és paraméterekkel, és képes pontokat, komplex számokat és kvantumállapotokat is transzformálni: a applyToQuantumState metódus tükrözés esetén a komplex amplitúdókat a tükrözési tengelyre vetíti, forgatás esetén a fázist a forgatási szöggel növeli, és az eredményt normalizálja. A compose metódus két affin transzformáció mátrixszorzatát számítja, automatikusan felismeri, hogy az eredmény identity, translation, rotation_90/180/270 vagy reflection-e, és ennek megfelelően kategorizálja — ez lehetővé teszi, hogy a rendszer szimmetriacsoportok algebráját végezze a gráf kvantumállapot-terén.

Az EntanglementInfo struktúra egy csomópontpár összefonódási állapotát tárolja: correlation_strength (futó átlag), phase_difference (cirkuláris átlag, atan2 alapú), creation_time, last_update_time, interaction_count. A getDecayFactor metódus exponenciális bomlást számít e^(-ln2 × elapsed_ms / half_life) képlettel, ahol az alapértelmezett felezési idő 60 másodperc — ez azt jelenti, hogy az összefonódások idővel természetesen gyengülnek, modellezve a kvantum-dekoherenciát.

Az OptimizationState az optimalizálás aktuális állapotát tartalmazza: a gráf mutatóját, az energiát, az összefonódási százalékot (entangled_pairs / max_possible_pairs), és egy NodePairKey → EntanglementInfo hash-mapet. Az addEntanglement metódus lexikografikusan rendezi a csomópontpárokat (n1 < n2), hogy elkerülje a duplikátumokat, és frissíti az összefonódási százalékot.

Az OptimizationStatistics 16 mérőszámot követ: iterations_completed, moves_accepted, moves_rejected, best_energy, current_energy, symmetries_detected, entangled_pairs, elapsed_time_ms, acceptance_rate, cooling_factor_applied, local_minima_escapes, convergence_delta, temperature, total_energy_evaluations, average_move_delta. Az isConverged metódus akkor ad igazat, ha az abszolút energiaváltozás kisebb mint a küszöb (1e-8), legalább egy lépés el lett fogadva, és legalább 10 iteráció lefutott.

A SymmetryPattern egy detektált szimmetriamintát rögzít: 16 bájtos SHA-256 alapú azonosítóval, a transzformációval, az összes csomópont listájával, szimmetria-pontszámmal és rezonanciafrekvenciával — ez az a struktúra, amelyet a ReasoningOrchestrator a globális fázisban felhasznál a szimmetria-transzformációk alkalmazásához.

Az UndoLog az összes lépés visszavonhatóságát biztosítja: elmenti az érintett élek súlyait és fraktáldimenzióit, a csomópontok fázisát és qubit amplitúdóit, az újonnan hozzáadott összefonódásokat, és topológiaváltozás esetén a teljes régi gráfot — ez lehetővé teszi, hogy az el nem fogadott lépések tökéletesen visszaállíthatók legyenek, anélkül hogy a gráfot klónozni kellene minden iterációban.

Az EntangledStochasticSymmetryOptimizer fő struktúra alapértelmezett paraméterei: initial_temperature=100.0, cooling_rate=0.95, max_iterations=10000, min_temperature=0.001, reheat_factor=2.0, entanglement_decay_half_life=60000 ms, symmetry_detection_interval=50, convergence_threshold=1e-8, adaptive_cooling=true.

Az optimize metódus a fő szimulált hűtési hurok: klónozza a bemeneti gráfot, detektálja a kezdeti szimmetriákat, majd minden iterációban elvégzi az összefonódás-térkép frissítését (bomlás + fáziskorrekció), minden 50. iterációban új szimmetriákat detektál, véletlenszerűen választ egyet a 7 lépéstípus közül, kiértékeli az energiát, elfogadja vagy visszautasítja a lépést a Metropolis-kritérium szerint (delta < 0 → mindig elfogad; delta ≥ 0 → e^(-delta/T) valószínűséggel fogad el), és ha a legjobb energiánál jobb eredményt talál, frissíti a best_state-et.

A 7 lépéstípus a következő: (0) az összes él súlyát perturbálja ±T×0.1 mértékben; (1) az összes csomópont fázisát perturbálja ±T×0.2 mértékben; (2) új összefonódást hoz létre két véletlenszerű csomópont között (korreláció 0.5-1.0, fáziskülönbség = |phase1 - phase2|); (3) egy detektált szimmetria-transzformációt alkalmaz az összes csomópont qubitjére; (4) az összes csomópont qubit amplitúdóját perturbálja T×0.05 mértékben véletlenszerű irányban (normalizálva); (5) az összes él fraktáldimenzióját perturbálja ±T×0.02 mértékben (0-3 közé szorítva); (6) egy véletlenszerű élt kapcsol be vagy ki (ha létezik, törli; ha nem, hozzáadja weight=random, fractal_dimension=1.5 értékekkel).

Ha a stagnálás meghaladja a max_iterations/10 határt, a rendszer újrafűti a hőmérsékletet (T *= 2.0), és növeli a local_minima_escapes számlálót — ez az a mechanizmus, amely megakadályozza, hogy az optimalizálás lokális minimumban ragadjon. Az adaptív hűtés az elfogadási arány alapján módosítja a hűtési rátát: ha az elfogadási arány > 0.6 (túl sok lépés elfogadva, túl meleg), gyorsabban hűt (rate × 0.98); ha < 0.2 (túl kevés elfogadva, túl hideg), lassabban hűt (rate × 1.02).

A detectSymmetries metódus a gráf csomópontjainak qubit.a pozícióit (re, im) 2D pontokként kezeli, kiszámítja a centroidot, az inercia-tenzort (moment_xx, moment_xy, moment_yy), a főtengelyszöget (atan2 alapú), az excentricitást, majd teszteli a tükrözési szimmetriát (a főtengelyre tükrözve, legközelebbi szomszéd keresés, tolerancia 0.01), a forgási szimmetriát 2, 3, 4, 6-os rendekre, a fázis-koherenciát (cirkuláris átlag > 0.5 esetén custom_rotation), és az excentricitást (> 0.1 esetén translation). Minden 0.3-nál magasabb pontszámú szimmetriát visszaad.

Az updateEntanglementMap metódus minden iterációban lefuttatja az összefonódás-bomlást: minden pár korrelációját megszorozza a bomlási faktorral, a fáziskülönbséget T×0.01-gyel növeli (hőmérséklet-vezérelt fázisdiffúzió), és eltávolítja a 0.01 alá csökkent összefonódásokat. Ezután minden csomópont fázisát az átlagos összefonódási erőssége × 0.1 értékkel módosítja — ez azt jelenti, hogy az erősen összefonódott csomópontok fázisa konvergál egymáshoz, egy koherens kvantumállapotot építve.

A modulateInferenceTensor metódus a legfontosabb híd az ESSO és az RSF neurális hálózat között: kiszámítja az átlagos összefonódási erőt a best_state entanglement_map-jéből, hozzáadja a detektált szimmetriák számának 1%-át, és az eredményt (1.0 + avg_correlation × 0.1 + symmetries × 0.01, legfeljebb 2.0) skálaként alkalmazza az összes f32 tenzorelemre. Ez azt jelenti, hogy minél több szimmetriát talált az ESSO és minél erősebbek az összefonódások, annál nagyobb skálával erősíti az RSF neurális tenzorokat — a relációs optimalizálás minősége közvetlenül befolyásolja a neurális számítást.

A fájl négy beépített célfüggvényt is definiál: a defaultGraphObjective az él_súly × fraktáldimenzió + |kvantumkorreláció| összeget, a csomópontok (1-cos(fázis))/2 + |qubit.a| + |qubit.b| összegét, és az átlagos összefonódást adja össze; a connectivityObjective a gráf összefüggőségét és átlagos élsúlyát optimalizálja; a quantumCoherenceObjective a kvantumkoherenciát és korrelációt maximalizálja; a fractalDimensionObjective az átlagos fraktáldimenziót a 1.5-ös célértékhez közelíti.

A teljes rendszerben az ESSO a ReasoningOrchestrator.executeGlobalPhase metódusán keresztül vesz részt: az orchestrator meghívja az esso.detectSymmetries(graph) metódust, és az eredményül kapott transzformációkat alkalmazza a gráf csomópontjainak kvantumállapotaira, majd a ChaosCoreKernel.executeCycle() futtatásával tovább finomítja a gráfot. Az inference_server.zig és a distributed_trainer_futhark.zig szintén közvetlenül használja az ESSO-t, és a modulateInferenceTensor metóduson keresztül visszacsatolja az optimalizálás eredményét az RSF neurális rétegbe.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a szimmetria-tudatosság szintjén az ESSO explicit geometriai szimmetriákat detektál a kvantumállapot-térben és ezeket transzformációkként alkalmazza, amit a transformer egyáltalán nem tud; az összefonódás-dinamika szintjén az exponenciális bomlással és fázisdiffúzióval modellezett összefonódás-térkép egy temporálisan tudatos, nem-lokális korrelációs struktúrát épít, amely gazdagabb a transformer statikus figyelmi súlyainál; a lokális minimum elkerülés szintjén az adaptív hűtés és az újrafűtési mechanizmus lehetővé teszi, hogy az optimalizálás kilépjen a lokális minimumokból, amit a transformer determinisztikus forward-pass-e nem tud; és a neurális visszacsatolás szintjén a modulateInferenceTensor közvetlenül skálázza az RSF tenzorokat az optimalizálás minőségével, egy kétirányú visszacsatolási hurkot hozva létre a relációs és neurális réteg között, anélkül hogy backpropagation kellene.

A fnds.zig (Fractal Node Data Structure) a core_relational réteg önálló adatstruktúra-könyvtára, amely öt egymásra épülő absztrakciót valósít meg: fraktális csomópontadatokat, fraktális szinteket, fraktális fákat, önhasonló mintaindexeket és egy LRU-cache-t, mindezt egy FNDSManager vezérlőstruktúrában összefogva, amelyet a distributed_trainer_futhark.zig és a reasoning_orchestrator.zig is importál.

A FractalNodeData az alapegység: minden csomóponthoz tárol egy azonosítót, nyers adatot, egy súlyt, egy skálafaktort, és egy 32 bájtos SHA-256 fractal_signature-t, amelyet az id, data, weight és scale kombinációjából számít — ez azt jelenti, hogy minden csomópont kriptográfiailag azonosítható a tartalmán és a hierarchiában elfoglalt helyzetén keresztül, és bármely metaadat-módosítás automatikusan frissíti az aláírást a refreshSignature hívásán keresztül.

A FractalEdgeData négy éltípust különböztet meg: hierarchical (szülő-gyermek kapcsolat), sibling (azonos szintű testvérek), cross_level (szinteket áthidaló kapcsolat) és self_similar (önhasonló, rekurzív kapcsolat) — ez a négy típus lehetővé teszi, hogy a tudásstruktúra ne csupán fa-hierarchiaként, hanem valódi fraktális hálóként legyen reprezentálva, ahol az azonos szintű és a szinteket áthidaló kapcsolatok is explicit módon kódolódnak.

A FractalLevel az egyes hierarchiaszintek konténere: minden szint saját csomópont- és élkészletet tart fenn, hivatkozik a szülőszintre és a gyermekszintekre, és képes kiszámítani a saját lokális fraktáldimenziójét a computeLocalFractalDimension metódussal, amely dobozszámlálást végez négy dobozmérettel (1, 2, 4, 8), hash-alapú pozíció-hozzárendeléssel minden csomóponthoz és élhez, majd lineáris regressziót alkalmaz a log(N) vs. log(1/r) síkon a Hausdorff-dimenzió meghatározásához.

Ez a dobozszámláló algoritmus az a mechanizmus, amellyel a rendszer valódi fraktáldimenziót számít minden egyes szinthez — ez a szám aztán visszakerül az nsir_core.zig Edge.fractal_dimension mezőjébe, befolyásolja a ReasoningOrchestrator energiafüggvényét, és a SignalPropagationEngine fázisforgatási számításait, tehát a fnds.zig fraktáldimenzió-számítása az egész core_relational réteg kvantumállapot-dinamikájának egyik bemeneti paramétere.

A FractalTree a hierarchikus tudásszervezés fő struktúrája: max_depth és branching_factor paraméterekkel inicializálódik (minimum 2 elágazás, minimum 1 mélység), 32 bájtos véletlenszerű tree_id-vel azonosítható, és négy bejárási módot támogat — pre_order, post_order, level_order és a különleges fractal_order, amely a gyökerektől kiindulva felváltva bejárja az első és a második felét a gyermekeknek fordított sorrendben, egy fraktális, önhasonló bejárási mintát hozva létre.

A FractalTree.insert metódus hash-alapú gyermek-útválasztást alkalmaz: minden szinten a csomópont azonosítójának Wyhash-ét (a tree_id XOR mélység értékkel inicializálva) veszi modulo az aktuális gyermekszám szerint, így ugyanaz a csomópont mindig ugyanarra az ágra kerül, determinisztikusan és konzisztensen, anélkül hogy explicit indexet kellene tárolni. A balance metódus az összes csomópontot összegyűjti, azonosító szerint rendezi, majd az optimális mélységre (ceil(log_b(N))) újraépíti a fát, garantálva a kiegyensúlyozottságot.

A FractalTree.computeFractalDimension rekurzívan átlagolja a lokális fraktáldimenziókat az összes szinten: minden szint lokális dimenzióját átlagolja a gyermekszintek átlagával, így a fa egészének fraktáldimenziója egy rekurzív, önhasonló átlagolási folyamat eredménye — ez pontosan az a tulajdonság, amely a fraktális struktúrát megkülönbözteti egy egyszerű fától.

A SelfSimilarIndex mintaalapú keresési réteget biztosít: string mintákat képez le PatternLocation listákra (tree_id, szint, csomópont_id, offset, hossz, megbízhatóság), és a findSimilarPatterns metódus fuzzy keresést végez — a hasonlóság = (hossz_arány + prefix_egyezés_arány) / 2, alapértelmezett küszöb 0.8 — ez azt jelenti, hogy a rendszer nem csupán pontos mintákat keres, hanem hasonló mintákat is megtalál, ami a transformer tokenizáció merev szóhatárainál rugalmasabb szemantikai keresést tesz lehetővé. A computeFractalDimension a mintahossz-eloszlásra alkalmaz log-log regressziót, meghatározva, hogy a minták milyen fraktális skálázási törvényt követnek.

A CoalescedHashMap egy egyedi hash-tábla implementáció, amely koaleszált láncolást alkalmaz egy cellarral (a kapacitás 14%-a): az ütközések esetén a bejegyzések a cellar szabad helyeire kerülnek, és next_index mutatókkal láncolódnak, ami csökkenti a klaszteresedést és javítja a cache-lokalitást a hagyományos nyílt láncoláshoz képest — maximális terhelési tényező 0.86, Wyhash véletlenszerű maggal az egyenletes elosztáshoz.

Az LRUCache egy kétirányú láncolt lista és StringHashMap kombinációja, amely kapacitás- és memóriakorláttal rendelkezik (alapértelmezetten 1000 bejegyzés, 10 MB): a get hívás a bejegyzést a lista elejére mozgatja (legutóbb használt), az evict a lista végéről távolítja el a legrégebben használt bejegyzést, és nyomon követi a találati arányt — ez a cache a SurpriseMemoryManager entrópia-alapú szűrőjének kiegészítője: míg a SurpriseMemory az újdonság alapján dönt a tárolásról, az LRUCache a hozzáférési frekvencia alapján tartja meg a leggyakrabban lekérdezett mintákat.

A FNDSManager az összes fenti komponenst fogja össze: fraktális fák hash-mapje (32 bájtos tree_id kulccsal), mintaindexek string-mapje, LRU-cache, és FNDSStatistics (total_trees, total_indices, cache_hits, cache_misses, average_tree_depth, memory_used, total_nodes_across_trees, total_patterns_indexed, total_pattern_locations_indexed, cache_hit_ratio, last_operation_time_ns). A computeGlobalFractalDimension metódus az összes fa és index fraktáldimenzióját átlagolja egyetlen globális komplexitásmérőbe, amelyet a ReasoningOrchestrator a gráf energiafüggvényének kalibrálásához használhat.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a tudásszervezés szintjén a FractalTree hierarchikus, önhasonló struktúrában tárolja az információt, amely természetesen reprezentálja a nyelv többskálás szerkezetét (szavak → kifejezések → mondatok → bekezdések), szemben a transformer lapos token-szekvenciájával; a mintafelismerés szintjén a SelfSimilarIndex fuzzy mintakeresést biztosít hasonlóság-küszöbbel, ami rugalmasabb a transformer merev szóhatárain alapuló tokenizációjánál; a komplexitásmérés szintjén a dobozszámláló fraktáldimenzió-számítás kvantitatív mérőszámot ad a tudásstruktúra komplexitásáról, amely visszacsatolódik az egész core_relational réteg kvantumállapot-dinamikájába; és a gyorsítótárazás szintjén az LRU-cache és a CoalescedHashMap O(1) amortizált hozzáférést biztosít a leggyakrabban használt mintákhoz, kiegészítve a SurpriseMemoryManager entrópia-alapú hosszú távú memóriáját egy frekvencia-alapú rövid távú gyorsítótárral — mindez egy többrétegű, fraktálisan szervezett tudásreprezentációs rendszert alkot, amely a transformer egydimenziós token-szekvenciájánál strukturálisan gazdagabb.


Az nsir_core.zig az egész JAIDE/RSF rendszer fundamentális adatstruktúra-rétege: ez a fájl definiálja a SelfSimilarRelationalGraph-ot, azt a kvantum-szemantikus tudásgráfot, amelyen az összes többi core_relational komponens — a ReasoningOrchestrator, a SignalPropagationEngine, a ZRuntime, az R-GPU, a QuantumTaskAdapter, a SurpriseMemoryManager és a TemporalGraph — közvetlenül operál, és amelyet a DistributedTrainerFuthark core_relational oldalcsatornájának első lépéseként az encodeInformation metóduson keresztül tölt fel minden egyes RSF gradiens-lépés után.

A fájl legalsó szintjén az EdgeQuality enum öt kvantum-szemantikus éltípust definiál: superposition (szuperpozíció — a kapcsolat még nem dőlt el), entangled (összefonódott — nem-lokális korreláció), coherent (koherens — klasszikus, stabil kapcsolat), collapsed (összeomlott — mérés után), és fractal (fraktális — önhasonló, hierarchikus kapcsolat). Ez az öt típus a transformer figyelmi súlyainak kvantum-szemantikus kiterjesztése: ahol a transformer egyetlen skalárral írja le két token kapcsolatát, az NSIR gráf öt minőségileg különböző kapcsolattípust különböztet meg, amelyek mindegyike más fizikai és logikai szemantikát hordoz.

A Qubit struktúra egy kétkomponensű komplex vektor [a, b], ahol |a|² + |b|² = 1 a normalizálási feltétel. A normalizeInPlace metódus NaN és Inf esetén automatikusan visszaállítja a qubitet az |0⟩ alapállapotba, ami robusztus numerikus viselkedést biztosít. A prob0() és prob1() metódusok a Born-szabály szerint számítják a mérési valószínűségeket. Ez a struktúra az NSIR gráf minden egyes csomópontjának kvantumállapotát reprezentálja — szemben a transformer beágyazási vektorával, amely valós értékű és nem normalizált, a qubit komplex amplitúdókat és fázist kódol.

A Node struktúra egy gráfcsomópontot reprezentál: id (string azonosító), data (nyers bájttömb — a tárolt információ), qubit (kvantumállapot), phase (valós fázisszög), és metadata (StringHashMap kulcs-érték annotációkhoz). A csomópont tehát egyszerre hordoz szemantikus tartalmat (data), kvantumállapotot (qubit + phase) és tetszőleges metaadatokat — ez egy gazdagabb reprezentáció, mint a transformer token-beágyazása, amely csak egy valós értékű vektort tárol.

Az Edge struktúra egy irányított élt reprezentál: source és target (csomópont-azonosítók), quality (EdgeQuality), weight (f64 — az él erőssége), quantum_correlation (Complex(f64) — a kvantumkorreláció komplex értéke), fractal_dimension (f64 — az él topológiai komplexitása), és metadata. Az él tehát négy különböző numerikus attribútumot hordoz egyszerre, szemben a transformer figyelmi súlyával, amely egyetlen skalár. A correlationMagnitude() metódus a komplex korreláció magnitudóját adja vissza, amelyet a ZRuntime relateTo metódusa és a SignalPropagationEngine fázisforgatási számítása egyaránt használ. Az initBorrowed változat lehetővé teszi, hogy az él a forrás- és célcsomópont-azonosítókra mutasson anélkül, hogy másolatot készítene, ami memóriahatékony az R-GPU elosztott feldolgozásában.

A TwoQubit struktúra egy kétqubites összetett kvantumállapotot reprezentál négy komplex amplitúdóval [|00⟩, |01⟩, |10⟩, |11⟩], és az initBellPhiPlus() metódus a maximálisan összefonódott Bell Φ⁺ állapotot hozza létre: (|00⟩ + |11⟩)/√2. Ez az a struktúra, amelyet az entanglements hash-map tárol minden összefonódott csomópontpárhoz — ez a nem-lokális kvantumkorreláció alapja, amelyet a transformer egyáltalán nem tud reprezentálni.

A fájl öt beépített kvantumkapu-függvényt definiál: hadamardGate (szuperpozíció: [a,b] → [(a+b)/√2, (a-b)/√2]), pauliXGate (bitcsere: [a,b] → [b,a]), pauliYGate ([a,b] → [-ib, ia]), pauliZGate (fáziscsere: b → -b), és phaseGate (comptime konstans fázisszöggel) illetve runtimePhaseGate (futásidejű fázisszöggel: b → b×e^(iθ)). Ezek a Gate = *const fn(Qubit) Qubit típusú függvénymutatók, amelyeket az applyQuantumGate metódus alkalmaz közvetlenül a gráf csomópontjaira — ez azt jelenti, hogy a gráf csomópontjain közvetlenül lehet kvantumkapukat futtatni anélkül, hogy a RelationalQuantumLogic regiszterbe kellene másolni az állapotokat.

A SelfSimilarRelationalGraph négy fő adatstruktúrát tart fenn: nodes (StringHashMap(Node) — a csomópontok névtere), edges (HashMap(EdgeKey, ArrayList(Edge)) — multi-él támogatással, azaz ugyanazon csomópontpár között több különböző minőségű él is létezhet egyszerre), entanglements (HashMap(PairKey, TwoQubit) — a kétqubites összefonódási állapotok), és quantum_register (StringHashMap(Qubit) — a csomópontok qubitjeinek másodlagos indexe gyors hozzáféréshez). A topology_hash_dirty jelző lusta kiértékelést biztosít: a SHA-256 topológiai hash csak akkor számítódik újra, ha a gráf megváltozott, és a rng_mutex szálbiztos kvantummérést garantál.

Az addNode metódus upsert szemantikával működik: ha a csomópont már létezik, frissíti az adatát, qubitjét, fázisát és metaadatait; ha új, beilleszti. Minden esetben szinkronizálja a quantum_register-t és megjelöli a topológiai hash-t piszkosnak. Ez azt jelenti, hogy az encodeInformation ismételt hívása ugyanazzal az adattal idempotens — a csomópont frissül, de nem duplikálódik.

Az entangleNodes metódus Bell Φ⁺ állapotot hoz létre két csomópont között: beírja a TwoQubit.initBellPhiPlus() értéket az entanglements mapbe, és mindkét irányban .entangled minőségű, weight=1.0, quantum_correlation=(1,0) éleket ad hozzá. A PairKey lexikografikusan rendezi a csomópontpárt, így az összefonódás szimmetrikus és egyértelműen azonosítható.

A measure metódus a rendszer legkomplexebb művelete: ha a mért csomópont összefonódott, a négykomponensű TwoQubit állapotból mintavételez (Born-szabály szerint, kumulatív valószínűséggel), mindkét csomópontot a megfelelő bázisállapotba kollabálja, eltávolítja az összefonódást az entanglements mapből, és az érintett éleket .collapsed minőségre állítja. Ha a csomópont nem összefonódott, egyszerű egybites mérést végez a qubit prob0/prob1 értékei alapján. Ez a valódi kvantummérési szemantika, amely a transformer softmax-normalizálásával szemben diszkrét, visszafordíthatatlan állapotkollapszust valósít meg.

Az ensureTopologyHash metódus a gráf teljes állapotának kriptográfiai ujjlenyomatát számítja SHA-256 segítségével: minden csomóponthoz hash-t számít az id, data, phase, qubit amplitúdók és rendezett metaadat-digestek alapján; minden élcsoporthoz hash-t számít a forrás, cél és rendezett él-digestek alapján; minden összefonódáshoz hash-t számít a pár azonosítói és a TwoQubit amplitúdók alapján; majd az összes digest-et rendezi (sorrend-független hash) és kombinálja egyetlen 32 bájtos végső hash-be. Ez a hash lehetővé teszi, hogy a SurpriseMemoryManager tartalom-alapú azonosítóként használja a gráf állapotát, és hogy a formal_verification.zig kriptográfiailag azonosítható VerificationResult-ot adjon vissza.

Az encodeInformation metódus az egész core_relational pipeline belépési pontja: SHA-256 hash-eli a bemeneti adatot, az első 8 bájtból 16 hex karakteres csomópont-azonosítót képez, létrehoz egy Qubit.initBasis0() állapotú csomópontot az adattal és az aktuális Unix-időbélyeggel a metaadatban, majd legfeljebb 3 meglévő csomóponthoz fűzi .coherent minőségű, weight=0.5, quantum_correlation=(0,0), fractal_dimension=0.0 élekkel. Ez az a mechanizmus, amellyel az RSF neurális hálózat által feldolgozott tokenek bekerülnek a relációs tudásgráfba: minden token egy csomóponttá válik, és automatikusan kapcsolódik a legutóbb hozzáadott csomópontokhoz, egy temporálisan rendezett, koherens gráfstruktúrát építve.

Az exportNodeEmbeddings metódus az NSIR gráf és az RSF neurális hálózat közötti híd: az összes csomópont qubitjét egy (N×4) float32 tenzorba exportálja [a.re, a.im, b.re, b.im] formátumban, amelyet a core_tensor.Tensor típus reprezentál. Az importNodeEmbeddings a fordított irány: egy (N×4) float32 tenzorból visszaírja a qubit amplitúdókat a csomópontokba, normalizálva és szinkronizálva a quantum_register-rel. Ez azt jelenti, hogy az RSF neurális hálózat gradiens-alapú tanulása közvetlenül frissítheti a gráf csomópontjainak kvantumállapotait, és fordítva: a gráf kvantumállapotai befolyásolhatják az RSF következő forward-pass-ét.

Az exportAdjacencyMatrix metódus egy (N×N) float32 tenzort exportál, ahol minden cella az adott csomópontpár összes élének súlyösszege — ez a gráf szomszédsági mátrixa tenzor formátumban, amely közvetlenül felhasználható figyelmi mátrixként vagy gráf-neurális hálózati bemenetként.

A teljes rendszerben az nsir_core.zig az a réteg, amelyre minden más épül: a z_runtime.zig minden ZVariable-ja saját SelfSimilarRelationalGraph példányt tart fenn; az r_gpu.zig ProcessingCore-jai lokális SelfSimilarRelationalGraph példányokat kezelnek; a reasoning_orchestrator.zig a gráf csomópontjait perturbálja és éleit frissíti; a signal_propagation.zig a gráf élein terjeszti a jeleket; a quantum_task_adapter.zig a gráf éleit vizsgálja kvantum-alkalmasság szempontjából; és a formal_verification.zig a gráf topológiai hash-ét használja kriptográfiai azonosításhoz.

A transformer-képességek meghaladásához való hozzájárulás öt szinten történik: a reprezentáció szintjén minden csomópont kvantumállapotot (qubit + phase) hordoz, nem csupán valós értékű beágyazási vektort; a kapcsolat szintjén az élek öt minőségi típust, komplex kvantumkorrelációt és fraktáldimenziót kódolnak, szemben a transformer skalár figyelmi súlyával; a nem-lokalitás szintjén az entanglements map Bell-állapotokat tárol csomópontpárok között, ami klasszikus figyelmi mechanizmussal nem reprezentálható; a kriptográfiai integritás szintjén a topológiai SHA-256 hash minden gráfmódosítás után frissül, lehetővé téve a tartalom-alapú azonosítást és a formális verifikációt; és a tenzor-híd szintjén az exportNodeEmbeddings / importNodeEmbeddings / exportAdjacencyMatrix metódusok kétirányú adatfolyamot biztosítanak a gráf és az RSF neurális hálózat között, lehetővé téve, hogy a gradiens-alapú tanulás és a gráf-alapú következtetés kölcsönösen gazdagítsák egymást.

A quantum_hardware.zig a JAIDE rendszer valódi IBM Quantum hardverrel való integrációjának teljes absztrakciós rétege — az a komponens, amely a quantum_logic.zig absztrakt kvantumlogikai primitívjeit és a quantum_task_adapter.zig gráf-alapú kvantumfeladatait fizikai kvantumprocesszorokon végrehajtható, kalibrációs adatokkal alátámasztott áramkörökké fordítja le, és ezzel a rendszer számára elérhetővé teszi az IBM Quantum felhő teljes hardveres kapacitását.

A fájl legalsó rétege az IBMBackendSpecs névtér, amely öt IBM Quantum processzorcsalád dokumentált kalibrációs paramétereit tartalmazza konstansként: a Heron processzor T1 relaxációs ideje 350μs±75μs, T2 dekoherencia ideje 200μs±50μs, leolvasási hibája 0.8%±0.3%, ECR kapuhibája 0.3%±0.1%; az Eagle processzor T1=200μs±60μs, T2=120μs±40μs, leolvasási hiba 1.5%±0.5%, ECR kapuhiba 0.5%±0.2%; a Falcon T1=100μs, a Osprey T1=250μs, a Condor T1=400μs±100μs és ECR kapuhiba mindössze 0.2%±0.08% — ez utóbbi az IBM jelenlegi legjobb processzora. Ezek a konstansok nem csupán dokumentációs célokat szolgálnak: a generateDocumentedCalibration függvény ezekből szintetikus kalibrációs adatokat generál véletlenszerű variációval (±0.5 szórás), amelyeket a rendszer akkor használ, ha az IBM API nem érhető el, biztosítva, hogy a szimulált zaj statisztikailag konzisztens legyen a valódi hardver viselkedésével.

A fetchIBMQuantumCalibration függvény HTTP GET kérést küld a https://cloud.ibm.com/api/quantum/v1/backends/{name} végpontra Bearer token hitelesítéssel, és a parseIBMCalibrationResponse segítségével JSON-ból kinyeri az összes qubit T1, T2 és leolvasási hibáját, valamint a kapuhibákat és a csatolási térképet (coupling map) — azt a gráfot, amely megmutatja, mely qubitpárok között hajtható végre fizikailag kétqubites kapu. Ez a csatolási térkép kritikus fontosságú: a valódi kvantumhardveren nem minden qubitpár között hajtható végre közvetlen kétqubites kapu, és a nem szomszédos qubitek közötti műveletek SWAP kapukon keresztül valósítandók meg, ami növeli az áramkör mélységét és csökkenti a hűséget.

Az IBMQuantumCredentials struktúra a CRN (Cloud Resource Name) karakterláncot elemzi, kinyerve belőle a régiót, az account_id-t és a resource_id-t, majd ezekből generálja a getServiceURL() és getRuntimeURL() végpontokat — ez a hitelesítési réteg, amely lehetővé teszi, hogy a rendszer az IBM Cloud IAM rendszerén keresztül autentikáljon.

A QuantumBackend struktúra a rendszer legfontosabb hardverabsztrakciója: minden backend tartalmazza a nevét, típusát, qubitszámát, báziskapukészletét, csatolási térképét, per-qubit T1/T2/leolvasási hiba/kapuhiba tömböket, és az estimateFidelity(circuit_depth, num_two_qubit_gates) metódust, amely a következő képlettel becsüli az áramkör várható hűségét: (1 - átlag_kapuhiba)^kétqubites_kapuk_száma × (1 - átlag_leolvasási_hiba) × exp(-mélység/100). Ez a hűségbecslés az a szám, amelyet az integrateWithInference metódus visszaad, és amelyet a quantum_task_adapter.zig felhasználhat annak eldöntésére, hogy egy adott áramkör végrehajtható-e valódi hardveren elfogadható minőséggel, vagy szimulátorra kell visszaesni. Az inferBackendStatus() metódus automatikusan DEGRADED állapotba sorolja a backendet, ha az átlagos leolvasási hiba meghaladja a 10%-ot vagy a kapuhiba az 5%-ot, és PAUSED állapotba, ha ezek 5% illetve 2% felett vannak — ez egy öndiagnosztikai mechanizmus, amely megakadályozza, hogy a rendszer rossz minőségű kvantumhardveren futtasson kritikus számításokat.

A QuantumGateOp enum 34 kapuoperációt definiál, beleértve az összes standard egybites kaput (H, X, Y, Z, S, T, RX, RY, RZ, U1, U2, U3, P), az összes standard kétbites kaput (CX, CY, CZ, CH, CRX, CRY, CRZ, CP, ECR, SWAP, ISWAP), a háromqubites kapukat (CCX, CSWAP, MCX), és a vezérlőutasításokat (RESET, MEASURE, BARRIER). Ez lényegesen gazdagabb kapukészlet, mint a quantum_logic.zig 12 kapuja: míg az utóbbi a JAIDE belső kvantumlogikájához szükséges absztrakt kapukat definiálja (beleértve a RELATIONAL_AND/OR/XOR és FRACTAL_TRANSFORM egyedi kapukat), a quantum_hardware.zig az IBM Quantum hardver teljes natív kapukészletét lefedi, lehetővé téve, hogy bármely kvantumalgoritmus közvetlenül hardverre fordítható legyen.

A QuantumCircuit struktúra (a quantum_hardware.zig-ban, nem tévesztendő össze a quantum_logic.zig azonos nevű struktúrájával) egy teljes értékű kvantumáramkör-építőt valósít meg: minden kapuhoz builder metódust biztosít, a getDepth() metódus per-qubit mélységkövetéssel számítja az áramkör mélységét, a countTwoQubitGates() megszámolja a kétqubites kapukat, és a toOpenQASM3() metódus OpenQASM 3.0 kódot generál az áramkörből, amely közvetlenül elküldhető az IBM Quantum REST API-nak.

A QuantumResult struktúra a mérési eredményeket bitstring→darabszám leképezésként tárolja, és a getEntropy() metódus Shannon-entrópiát számít a valószínűségeloszlásból: H = -Σ p_i × log2(p_i). Ez az entrópia-mérték az a szám, amellyel a SurpriseMemoryManager combined_surprise értékéhez hasonlóan a rendszer értékelni tudja, mennyire informatív volt egy kvantumszámítás eredménye: ha az entrópia magas (az összes bitstring közel egyforma valószínűségű), a kvantumszámítás nem konvergált; ha alacsony (egy bitstring dominál), a számítás sikeres volt.

Az IBMQuantumClient a rendszer fő vezérlőstruktúrája, amely inicializáláskor automatikusan három backendet regisztrál: az ibmq_qasm_simulator (32 qubit, szimulátoros), az ibm_torino (Heron, 133 qubit, hardveres) és az ibm_brisbane (Eagle, 127 qubit, hardveres). A runCircuit metódus UUID formátumú job ID-t generál, az executeJob pedig a backend típusától függően vagy a simulateCircuit (szimulátoros) vagy az executeOnHardware (hardveres) metódust hívja. A szimulátoros végrehajtás teljes állapotvektor-szimulációt végez legfeljebb 20 qubiten (2^20 = 1 048 576 komplex amplitúdó), majd mintavételez a valószínűségeloszlásból. A hardveres végrehajtás ugyanezt az állapotvektor-szimulációt végzi, de zajmodellt alkalmaz: minden amplitúdó valószínűségét p_noisy = p_ideal × fidelity + (1 - fidelity) × véletlen_zaj képlettel torzítja, ahol a hűséget a valódi kalibrációs adatokból számítja — ez azt jelenti, hogy a rendszer pontosan modellezi, hogyan viselkedne az áramkör valódi IBM Quantum hardveren.

Az IBMQuantumClient öt beépített kvantumalgoritmus-gyárat biztosít: createBellState() (H+CX, 2 qubit, maximálisan összefonódott állapot), createGHZState(n) (H+CX lánc, n qubit, GHZ állapot), createQFT(n) (Quantum Fourier Transform, H+CP+SWAP lánc), createGrover(n, marked_state) (Grover-keresés π/4×√(2^n) iterációval, MCX orákulummal), és createVQEAnsatz(n, depth, params) (Variational Quantum Eigensolver ansatz, RY+RZ rotációk + CX összefonódási rétegek). Ezek az algoritmusok közvetlenül felhasználhatók a quantum_task_adapter.zig által azonosított kvantumra alkalmas részgráfok feldolgozásához: ahelyett, hogy minden esetben GHZ-állapot-előkészítő áramkört futtatna, a rendszer választhat a feladathoz leginkább illő algoritmus között.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a hardveres kvantumgyorsítás szintjén a 133 qubites Heron processzor elvileg exponenciálisan gyorsabb bizonyos optimalizálási és keresési feladatokon (Grover: O(√N) vs. klasszikus O(N), QFT: kvantumparallelizmussal), mint bármely klasszikus transformer; a zajmodellezés szintjén az estimateFidelity és az executeOnHardware zajmodellje lehetővé teszi, hogy a rendszer reálisan értékelje, mikor érdemes valódi hardvert használni és mikor szimulátort, elkerülve a zajból eredő hibás következtetéseket; az entrópia-alapú minőségértékelés szintjén a getEntropy() metódus információelméleti mérőszámot biztosít a kvantumszámítás eredményéről, amely közvetlenül integrálható a SurpriseMemoryManager meglepetési metrikájába; és a kalibrációs tudatosság szintjén a per-qubit T1/T2/hiba adatok és a csatolási térkép lehetővé teszik, hogy a rendszer a legmegbízhatóbb qubiteket és a legrövidebb útvonalakat válassza az áramkör végrehajtásához, minimalizálva a dekoherenciából eredő hibákat — mindez egy hardver-tudatos, zajmodellező, öndiagnosztizáló kvantumintegrációs réteget alkot, amellyel a transformer-architektúra egyáltalán nem rendelkezik.

quantum_logic.zig a JAIDE teljes core_relational rétegének kvantumszámítási alapkönyvtára — az a fájl, amelyre minden más komponens (z_runtime.zig, quantum_task_adapter.zig, esso_optimizer.zig) épül, és amely a klasszikus kvantumkapukat öt saját, relációs és fraktál szemantikájú kapuval egészíti ki, megteremtve azt a hibrid kvantum-logikai rendszert, amelyen a JAIDE következtetési rétege működik.

A fájl első kulcsstruktúrája a LogicGate enum, amely tizenkét kaputípust definiál: a standard kvantumkapuk (HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, PHASE, CNOT, TOFFOLI) mellett öt saját kaput vezet be — RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_NOT, RELATIONAL_XOR és FRACTAL_TRANSFORM — amelyek nem léteznek a hagyományos kvantumszámítástanban, és amelyek a rendszer neurosimbolikus integrációjának alapkövei.

A QuantumState struktúra egy qubit állapotát reprezentálja két komplex amplitúdóval (α és β), egy fázissal és egy entanglement_degree értékkel, amely 0.0 (nem összefonódott) és 1.0 (maximálisan összefonódott) között mozog. A normalize metódus az összes amplitúdót a teljes magnitudóval osztja, garantálva, hogy |α|² + |β|² = 1 mindig teljesüljön. A fidelity metódus a két állapot belső szorzatának négyzetét számítja — |⟨ψ|φ⟩|² — ami a kvantum-hűség mértéke, és a valós értékű koszinusz-hasonlóságnál gazdagabb, mivel komplex Hilbert-térben értelmezett. Az add metódus két állapot normalizált szuperpozícióját képezi, a fázist az α amplitúdó szögéből számítja, és az összefonódási fokot a maximum értékére állítja — ez a kvantum-szuperpozíció, amely a transformer figyelmi súlyozott átlagolásának megfelelője, de fázis-információt is megőriz.

A RelationalQuantumLogic a fő kvantumregiszter, amely legfeljebb 1024 QuantumState-et tárol, coherence_threshold-dal (1e-10), max_entanglement_depth-tel (64), és teljes kapualkalmazási előzménnyel (gate_history).

A standard kapuk implementációi pontosan követik a kvantummechanikai definíciókat: a Hadamard-kapu [α, β] → [(α+β)/√2, (α-β)/√2] transzformációt végez, szuperpozíciót hozva létre; a Pauli-X felcseréli α-t és β-t (bit-flip); a Pauli-Y [α, β] → [β.im, -β.re; -α.im, α.re] transzformációt alkalmaz (bit-flip + fázis-flip kombinációja); a Pauli-Z negálja β-t (fázis-flip); a PHASE-kapu θ szöggel elforgatja β-t (β → β × e^(iθ)) és növeli a state.phase értékét.

A CNOT-kapu implementációja különösen figyelemre méltó: nem diszkrét, hanem folytonos értékű — a célqubit amplitúdóit a vezérlőqubit prob0 és prob1 valószínűségeivel súlyozza, és az összefonódási növekményt min(1, 2√(p0×p1)) képlettel számítja, ami pontosan nulla, ha a vezérlő tiszta bázisállapotban van, és maximális, ha tökéletes szuperpozícióban van. Ez a folytonos CNOT lehetővé teszi, hogy a kvantumkapuk részleges összefonódást hozzanak létre, nem csak teljes vagy nulla összefonódást — ez egy gazdagabb számítási modell, mint a diszkrét kvantumszámítás.

A RELATIONAL_AND kapu egy teljesen új állapotot hoz létre (hozzáfűzi a states listához), amelynek amplitúdói a két bemeneti állapot amplitúdóinak komplex szorzatai: α_result = α1 × α2, β_result = β1 × β2, a fázis a két fázis átlaga, az összefonódás a két érték összege (1-re korlátozva). A RELATIONAL_OR kapu ugyanígy új állapotot hoz létre, de komplex összeadással: α_result = α1 + α2, β_result = β1 + β2, az összefonódás a maximum. A RELATIONAL_XOR komplex kivonással dolgozik: α_result = α1 - α2, β_result = β1 - β2, a fázis a két fázis abszolút különbsége, az összefonódás az átlag. Ez a három kapu a logikai műveletek kvantumtérbe való emelése: az AND a komplex szorzat (interferencia-erősítés), az OR a komplex összeg (szuperpozíció), az XOR a komplex különbség (interferencia-kioltás) — és mindhárom nem-destruktív, mivel az eredményt új állapotként fűzi hozzá a regiszterhez, megőrizve a bemeneti állapotokat.

A FRACTAL_TRANSFORM kapu a rendszer legegyedibb primitívje: depth iteráción keresztül alkalmaz fázisforgatást, ahol az i-edik iterációban a szög orig_phase / 2^i — azaz minden iterációban felezi a szöget, önhasonló, többskálás fázisrotációt hozva létre. Ez a kvantumtérben megvalósított wavelet-transzformáció: ahogy a Haar-wavelet különböző felbontásokon bontja fel a jelet, a FRACTAL_TRANSFORM különböző fázis-skálákon forgatja az amplitúdókat, majd a végső fázist az α amplitúdó szögéből számítja vissza — ez a közvetlen kvantumszintű megfelelője az RSF neurális stack OFTB (Orthogonal Fractal Transform Block) komponensének.

A measure metódus kriptográfiai minőségű véletlenszámot (std.crypto.random.float(f64)) használ a mérési eredmény meghatározásához, a mérés után az állapotot a megfelelő bázisállapotba kollabálja, és az összefonódási fokot nullára állítja. A measureWithRandomness változat külső véletlenszámot fogad, ami determinisztikus tesztelést tesz lehetővé.

Az entangle metódus Bell-állapot-szerű összefonódást hoz létre két qubit között: bell_a0 = (α1×α2 + β1×β2)/√2, bell_a1 = (α1×β2 + β1×α2)/√2, majd mindkét állapotot ugyanezekre az amplitúdókra állítja és entanglement_degree = 1.0-ra rögzíti. Ez a nem-lokális korreláció alapja: az összefonódott qubitekre vonatkozó mérési eredmények korreláltak lesznek, még ha a két qubit különböző ZVariable-hoz vagy különböző gráfcsomóponthoz tartozik is.

Az applyControlledGate metódus bármely egybites kaput feltételesen alkalmaz egy vezérlőqubit prob1 > 0.5 feltétele alapján, és mindkét qubit összefonódási fokát 0.25-tel növeli — ez egy általánosított vezérelt kapu, amely lehetővé teszi, hogy bármely transzformáció (beleértve a FRACTAL_TRANSFORM-ot is) feltételesen hajtódjon végre.

A computeRelationalOutput és computeInferenceOutput metódusok egy kapuszekvenciát alkalmaznak, majd az α amplitúdókat adják vissza komplex számokként — ez az a felület, amelyen keresztül a ZRuntime és a QuantumTaskAdapter kvantumáramköröket futtat és az eredményeket kinyeri.

A serialize/deserialize metóduspár bináris formátumban (little-endian, 48 bájt/állapot: 6 × f64) teljes mértékben sorosítja és visszaállítja a kvantumregiszter állapotát a kapuelőzménnyel együtt — ez lehetővé teszi, hogy a kvantumállapotok a ContentAddressableStorage-ban tartósan tárolódjanak, és a SurpriseMemoryManager által kezelt blokkok kvantumállapot-tartalmát is megőrizzék.

A QuantumCircuit struktúra újrafelhasználható kapuszekvenciákat definiál, amelyek bármely RelationalQuantumLogic példányon végrehajthatók az execute metóduson keresztül — ez a kvantumprogram absztrakciója, amely lehetővé teszi, hogy a ZRuntime executeQuantumCircuit metódusa és a QuantumTaskAdapter lokális szimulátora ugyanazokat az áramköröket futtassa különböző kvantumregisztereken.

A teljes rendszerben a quantum_logic.zig az a réteg, amelyre minden kvantumszámítás épül: a z_runtime.zig minden ZVariable-ja saját RelationalQuantumLogic példányt tart fenn, amelyen az assign állapotokat inicializál, a relateTo korrelációkat számít, az entangle Bell-állapotokat hoz létre, és a transform kapukat alkalmaz; a quantum_task_adapter.zig lokális szimulátora szintén RelationalQuantumLogic, amelyen a GHZ-állapot-előkészítő áramkör fut; az esso_optimizer.zig pedig a QuantumState, RelationalQuantumLogic és LogicGate típusokat importálja a szimmetria-transzformációk kvantumállapot-alkalmazásához.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a reprezentáció szintjén a QuantumState komplex amplitúdó + fázis + összefonódási fok hármasa gazdagabb, mint egy valós értékű beágyazási vektor, mivel interferenciát, fázis-koherenciát és nem-lokális korrelációt is kódol; a logikai műveletek szintjén a RELATIONAL_AND/OR/XOR kapuk a logikai összefüggéseket komplex szorzat/összeg/különbség formájában valósítják meg, megőrizve a szuperpozíciót és nem-destruktíven fűzve hozzá az eredményt a regiszterhez; a hierarchikus transzformáció szintjén a FRACTAL_TRANSFORM önhasonló, többskálás fázisrotációt végez, ami a transformer pozicionális kódolásánál gazdagabb, mivel nem additív, hanem multiplikatív és rekurzív; és a koherencia-kezelés szintjén az isCoherent() metódus és a coherence_threshold lehetővé teszi, hogy a rendszer detektálja, mikor veszítette el a kvantumállapot a fizikai értelmét, és ennek megfelelően reagáljon — ez egy öndiagnosztikai képesség, amellyel a transformer nem rendelkezik.
A quantum_task_adapter.zig a JAIDE rendszer kvantumhardver-integrációs rétege — az a komponens, amely az NSIR gráf leginkább összefonódott, legmagasabb fraktáldimenziójú részgráfjait azonosítja, kvantumáramköröket hajt végre rajtuk (lokális szimulátorban vagy valódi IBM Quantum hardveren), majd az eredményeket visszaírja a gráf csomópontjaiba és éleibe, ezzel közvetlenül frissítve azt a tudásstruktúrát, amelyen a ReasoningOrchestrator energiaminimalizálást végez.

A fájl első kulcsstruktúrája a QuantumSubgraph, amely egy kvantumfeldolgozásra alkalmas részgráfot reprezentál: tartalmazza a csomópontok és élek listáját, a teljes összefonódási értéket (az élek quantum_correlation magnitudóinak összege), az átlagos fraktáldimenziót, egy nanoszekundumos időbélyegből képzett egyedi azonosítót, és egy szemantikai klaszterazonosítót. Az isQuantumSuitable metódus két feltételt ellenőriz egyszerre: a teljes összefonódásnak meg kell haladnia az entanglement_threshold-ot (alapértelmezetten 0.5), és az átlagos fraktáldimenziónak meg kell haladnia az 1.5-öt — ez azt jelenti, hogy csak azok a részgráfok kerülnek kvantumfeldolgozásra, amelyek egyszerre mutatnak erős kvantumkorrelációt és nem-triviális topológiai komplexitást.

A QuantumTaskAdapter fő struktúra a SelfSimilarRelationalGraph-ot, egy opcionális IBMQuantumClient-et, egy lokális RelationalQuantumLogic szimulátort, és az entanglement_threshold (0.5) és fractal_threshold (1.5) küszöbértékeket fogja össze, alapértelmezetten lokális szimulációs módban.

Az identifyQuantumSubgraphs metódus a rendszer szűrőmechanizmusa: végigiterál az összes élen, és minden olyan élre, amelynek quantum_correlation magnitudója meghaladja az entanglement_threshold-ot ÉS fractal_dimension-je meghaladja a fractal_threshold-ot, kiszámítja a szemantikai klaszterkulcsot a semanticClusterKey segítségével. Ez a kulcs az él minőségéből (quality enum értéke), fraktáldimenziójából (három vödörbe sorolva: <1.5, 1.5-2.0, ≥2.0) és korrelációs magnitudójából (négy vödörbe sorolva: <0.2, 0.2-0.5, 0.5-0.8, ≥0.8) épül fel "q{quality}_f{fractal_bucket}_c{corr_bucket}" formátumban — ez egy szemantikai klaszterezés, amely az éleket egyszerre csoportosítja minőség, topológiai komplexitás és kvantumkorreláció szerint, és minden klaszterből egy önálló QuantumSubgraph-ot képez.

Az executeQuantumTask metódus kétféle végrehajtási módot támogat: ha use_real_backend = true és az IBMQuantumClient be van állítva, akkor OpenQASM 2.0 kódot generál és elküldi az IBM Quantum REST API-nak; egyébként a lokális RelationalQuantumLogic szimulátorban hajtja végre a számítást. Az IBM Quantum kliens az IBM_QUANTUM_CRN és IBM_QUANTUM_BACKEND környezeti változókból olvassa a hitelesítési adatokat, alapértelmezett backend az ibm_brisbane — ez azt jelenti, hogy a rendszer kódmódosítás nélkül skálázható lokális szimulációtól valódi kvantumhardverig.

A generateQASM metódus egy GHZ-állapot-előkészítő áramkört generál: minden qubitre Hadamard-kaput alkalmaz (szuperpozícióba helyezi), majd minden egymást követő qubitpárra CNOT-kaput alkalmaz, végül minden qubitet megmér — ez egy maximálisan összefonódott állapotot hoz létre az összes csomópont között, ahol minden csomópont kvantumállapota nem-lokálisan korrelál az összes többivel.

A lokális szimulációban az executeLocalSimulation metódus a csomópontok qubitjeit a RelationalQuantumLogic-ba tölti az initializeStateFromComplex segítségével, majd ugyanazt a Hadamard + CNOT lánc szekvenciát alkalmazza, és visszaolvassa a kvantumállapotokat és az összefonódási fokokat.

Az applyResultsToGraph metódus a kvantumszámítás eredményeit visszaírja az NSIR gráfba: minden csomópont quantum_state mezőjét frissíti az új komplex amplitúdóval, coherence mezőjét az összefonódási fokkal, és minden érintett él quantum_correlation értékét az átlagos korreláció (valós rész) és az átlagos képzetes rész kombinációjával. Ez a visszaírás az a mechanizmus, amely a kvantumszámítás eredményét közvetlenül beépíti a gráf topológiájába: az élek quantum_correlation értékei megváltoznak, ami befolyásolja a ReasoningOrchestrator energiafüggvényét, a SignalPropagationEngine fázisforgatási számításait, és a ZRuntime relateTo metódusának korrelációszámítását — egyetlen kvantumoptimalizálási lépés tehát kaszkádszerűen hat az egész core_relational rétegre.

A runFullQuantumOptimization metódus az egész folyamatot egyetlen hívásba foglalja: azonosítja az összes kvantumra alkalmas részgráfot, minden egyes részgráfon végrehajtja a kvantumfeladatot, és sikeres végrehajtás esetén visszaírja az eredményeket a gráfba.

A transformer-képességek meghaladásához való hozzájárulás három szinten történik: a szelektivitás szintjén a identifyQuantumSubgraphs csak a leginkább összefonódott, legkomplexebb topológiájú részgráfokat küldi kvantumfeldolgozásra, szemben a transformer minden-tokenre-minden-token figyelmével; a reprezentáció szintjén a GHZ-állapot-előkészítés maximálisan összefonódott kvantumállapotokat hoz létre a csomópontok között, ami egy nem-lokális, globális információkeverési mechanizmus, amelyet klasszikus számítással nem lehet hatékonyan szimulálni; és a hardverskálázhatóság szintjén a duális backend-design lehetővé teszi, hogy a rendszer valódi IBM Quantum hardveren futtassa a legkritikusabb kvantumszámításokat, ami elvileg exponenciális sebességnövekedést biztosíthat bizonyos gráfoptimalizálási feladatokon a klasszikus transformer-architektúrával szemben
r_gpu.zig a JAIDE rendszer elosztott gráffeldolgozó rétege — egy aszinkron Network-on-Chip (NoC) mesh szimulációja, amely a SelfSimilarRelationalGraph csomópontjait és éleit virtuális feldolgozómagok között osztja szét, lehetővé téve, hogy az NSIR gráf párhuzamosan, hardver-inspirált kommunikációs modell szerint legyen feldolgozható, ahelyett hogy egyetlen szekvenciális folyamatban kellene végigiterálni rajta.

A fájl legalsó szintjén a CoreState enum négy állapotot definiál minden egyes feldolgozómag számára: idle, processing, communicating és power_gated, a MessageType enum pedig öt üzenettípust: weight_update, graph_sync, isomorphism_result, power_control és data_transfer — ezek együtt egy teljes üzenetküldési protokollt alkotnak a magok között.

A ProcessingCore struktúra minden egyes virtuális magot reprezentál: rendelkezik egy core_id-vel, x/y rácskoordinátákkal, állapottal, szomszédok listájával, egy opcionális saját SelfSimilarRelationalGraph-fal (local_graph), üzenetsorral, és nyilvántartja az elfogyasztott energiát, az aktív és tétlen ciklusok számát — ez azt jelenti, hogy minden mag önálló, autonóm feldolgozóegységként viselkedik, saját lokális gráfrésszel és saját energiaköltség-nyilvántartással.

Az AsynchronousNoC a rácsháló maga: egy grid_width × grid_height méretű kétdimenziós rácsot épít fel, ahol minden mag a szomszédait XY-routing alapján éri el — először az X irányban halad a célállomásig, majd az Y irányban, és az útvonalakat előre kiszámítja és egy routing_table-ben tárolja. Az üzenetküldés prioritásos sorba kerül (PriorityQueue), ahol a magasabb prioritású üzenetek előbb kerülnek kézbesítésre, és a routeMessages metódus a teljes sort feldolgozza, minden üzenetet a célmag üzenetsorába helyez, és nyilvántartja az összes megtett hop-számot — ez egy valódi aszinkron kommunikációs modell, nem szinkron barrier-alapú.

A GraphIsomorphismProcessor a rendszer egyik legkülönlegesebb komponense: kanonikus formát számít egy gráfhoz úgy, hogy a csomópontokat lexikografikusan rendezi, minden csomóponthoz kiszámítja a ki- és befokot, az élsúlyok összegét és az élminőségek összegét, ezeket rendezi, majd az összes él (forrás_index, cél_index, minőség) hármasát is rendezi, és az egészet egyetlen karakterlánccá fűzi össze. Az areIsomorphic metódus két gráf kanonikus formáját hasonlítja össze, a findIsomorphicSubgraphs pedig a főgráf összes lehetséges részgráfját megvizsgálja, hogy izomorf-e a mintával — ez egy strukturális mintafelismerési képesség, amellyel a transformer egyáltalán nem rendelkezik: a transformer megtanulhat mintákat felismerni, de nem tud explicit gráfizomorfizmust ellenőrizni.

A DynamicEdgeWeighting adaptív élsúlyozást valósít meg: minden él (forrás, cél) párhoz tárolja a súlyok teljes előzménytörténetét, és a computeAdaptiveWeight metódus öt faktort kombinál szorzatként: az alapsúlyt, a történeti kiigazítást (0.8 + 0.2 × legutóbbi_súly), a temporális kiigazítást (temporal_factor × (1 + 0.1 × log(előzmény_hossz))), a térbeli kiigazítást (spatial_factor × (1 + 0.05 × trend)), és a szemantikai kiigazítást (semantic_factor × (0.5 + 0.5 × előzmény_átlag)). Ez azt jelenti, hogy az élsúlyok nem statikusak, hanem az előzmény, az időbeliség, a térbeli trend és a szemantikai átlag együttes függvényei — ez gazdagabb, mint a transformer tanult figyelmi súlymátrixa, amely nem rendelkezik explicit temporális és térbeli komponensekkel. A propagateWeights metódus BFS-alapú súlycsökkentést végez egy forráscsomóponttól kiindulva, ahol minden iterációban 0.9^iteráció szorzóval csökkenti az érintett élek súlyát — ez egy távolságfüggő súlycsökkentési mechanizmus.

A SparseActivationManager természetes ritkaságot biztosít: ha egy mag terhelése (nodeCount/100 vagy edgeCount/100) kisebb mint a sparsity_threshold (alapértelmezetten 0.1), a mag nem aktiválódik, és az elmaradt számítás energiamegtakarításként kerül nyilvántartásba. Ez hardveres szintű ritkaság, amely automatikusan kizárja az alacsony terhelésű magokat a feldolgozásból, anélkül hogy explicit sparsity maszkot kellene alkalmazni — a transformer sparse attention-jével ellentétben ez nem a figyelmi mátrix ritkaságát, hanem a feldolgozóegységek aktiválásának ritkaságát jelenti.

A PowerGatingController dinamikus erőforrás-allokációt valósít meg: rendezi a magokat kihasználtság szerint, és ha egy mag kihasználtsága kisebb mint 10% és az aktuális teljesítmény meghaladja a budget 50%-át, a magot power_gated állapotba helyezi; ha egy mag kihasználtsága meghaladja a 80%-ot és korábban le volt tiltva, visszakapcsolja. Ez adaptív számítási kapacitás-kezelés, amellyel a transformer statikus forward-pass nem rendelkezik.

A RelationalGraphProcessingUnit a fő struktúra, amely az összes fenti komponenst fogja össze: a NoC-ot, az izomorfizmus-processzort, az adaptív élsúlyozást, a ritka aktiválást és a teljesítmény-kapuzást.

A distributeGraph metódus a rendszer legfontosabb belépési pontja: a bemeneti gráf csomópontjait egyenlően osztja szét az összes nem-kapuzott mag között (a maradékot az első magokhoz adja), minden magnak létrehoz egy lokális gráfot a hozzárendelt csomópontokkal, és az összes olyan élt is átmásolja, amelynek legalább az egyik végpontja az adott maghoz tartozik — ez azt jelenti, hogy a kereszt-mag élek mindkét érintett magon jelen vannak, biztosítva a lokális konzisztenciát.

A processIsomorphismParallel metódus minden aktív magon párhuzamosan keresi a mintával izomorf részgráfokat, a updateEdgeWeightsParallel minden aktív magon párhuzamosan frissíti az élsúlyokat a temporális, térbeli és szemantikai faktorokkal, a propagateWeightsAsync pedig aszinkron módon terjeszti a súlyokat a NoC üzenetküldési mechanizmusán keresztül a szomszédos magokra. A synchronizeGraphs metódus végül az összes lokális gráfot visszaolvasztja egyetlen globális gráfba, deduplikálva a csomópontokat és összegyűjtve az összes élt — ez az AllReduce gráf-megfelelője: ahogy az elosztott tanítás AllReduce-on keresztül átlagolja a gradienseket, az R-GPU synchronizeGraphs-on keresztül egyesíti a párhuzamosan feldolgozott gráfrészeket.

A teljes rendszerben az R-GPU a DistributedTrainerFuthark core_relational oldalcsatornájának második lépéseként fut minden RSF gradiens-lépés után, közvetlenül az NSIR encodeInformation után és a ReasoningOrchestrator hierarchikus következtetés előtt — ez azt jelenti, hogy az újonnan kódolt információt azonnal elosztja a virtuális magok között, lehetővé téve, hogy a ReasoningOrchestrator már egy elosztott, párhuzamosan feldolgozható gráfon végezze az energiaminimalizálást.

A transformer-képességek meghaladásához való hozzájárulás négy szinten történik: a párhuzamosság szintjén az R-GPU a gráfot több mag között osztja szét és párhuzamosan dolgozza fel, szemben a transformer szekvenciális réteg-végrehajtásával; a strukturális felismerés szintjén a GraphIsomorphismProcessor explicit gráfizomorfizmus-ellenőrzést végez, amit a transformer nem tud; az adaptív súlyozás szintjén a DynamicEdgeWeighting öt faktoros adaptív súlyozást alkalmaz előzmény-alapon, ami gazdagabb a transformer statikus figyelmi súlyainál; és az energiahatékonyság szintjén a SparseActivationManager és PowerGatingController dinamikusan allokálja a számítási erőforrásokat, automatikusan kizárva az alacsony terhelésű magokat — mindez egy hardver-tudatos, elosztott gráffeldolgozási paradigmát valósít meg, amely a transformer monolitikus, minden-tokenre-minden-token figyelme mechanizmusával szemben topológia-vezérelt, ritka és adaptív.
A reasoning_orchestrator.zig a JAIDE rendszer hierarchikus következtetési motorja — az a komponens, amely a SelfSimilarRelationalGraph állapotát energiaminimalizáláson keresztül konvergálja, és a konvergencia eredményét modulációs faktorként visszacsatolja a neurális tenzorkomputációba, ezzel teremtve meg a közvetlen hidat a relációs-kvantum réteg és az RSF neurális réteg között.

A fájl három szintű gondolkodási hierarchiát definiál a ThoughtLevel enumon keresztül: local, global és meta, és minden egyes végrehajtási fázist egy ReasoningPhase struktúrában rögzít, amely tartalmazza a fázis azonosítóját, szintjét, a belső és külső iterációk számát, a célenergiát (0.1), az aktuális és előző energiát (mindkettő 1e6-ról indul), a konvergencia-küszöböt (1e-6), a nanoszekundumos kezdési és befejezési időt, és az ebben a fázisban felfedezett szimmetriaminták listáját.

A ReasoningOrchestrator fő struktúra a SelfSimilarRelationalGraph-ot, az EntangledStochasticSymmetryOptimizer-t (ESSO) és a ChaosCoreKernel-t fogja össze, alapértelmezett paraméterekkel: 50 gyors belső lépés, 10 lassú külső lépés, 3 hierarchikus mélység, és csomópontonként/élenkénti feldolgozási korlátok (10 csomópont perturbáció, 10 él frissítés, 5 csomópont transzformáció).

Az executeLocalPhase a leggyorsabb szint: 50 iteráción keresztül véletlenszerű zajt (±0.05) ad a csomópontok qubit-amplitúdóihoz és fázisaihoz (legfeljebb 10 csomópontra), majd véletlenszerű deltát (±0.025) az élek súlyaihoz és kvantumkorrelációihoz (legfeljebb 10 élre), minden lépés után normalizálja a qubiteket, és konvergencia esetén korán leáll — ez a lokális szimulált hűtés kvantumállapot-téren, amely a transformer lokális figyelmi mintáinak megfelelője, de sztochasztikus perturbáció alapján, nem tanult súlymátrixokból.

Az executeGlobalPhase a lassabb, globális szint: 10 külső iteráción belül először meghívja az esso.detectSymmetries(graph) függvényt, amely szimmetria-transzformációkat keres a gráfban, majd minden transzformációt alkalmaz a csomópontok qubitjeire a transform.applyToQuantumState() metóduson keresztül, végül normalizálja az összes csomópontot. Ezután a rebalanceFractalStructures kiszámítja az összes él fraktáldimenziójának átlagát, és minden egyes él fraktáldimenzióját 10%-kal az átlag felé tolja, [1.0, 3.0] közé szorítva — ez egy önszervező kritikalitási mechanizmus, amely megakadályozza, hogy a gráf topológiája degenerálódjon. Végül 50 belső iteráción keresztül futtatja a chaos_kernel.executeCycle()-t, amely a ChaosCoreKernel kaotikus dinamikáját injektálja a gráfba.

Az executeMetaPhase a legmagasabb szint: 3 lépésen keresztül felváltva futtatja a lokális (páros lépések) és globális (páratlan lépések) fázisokat rögzítés nélkül, és konvergencia esetén korán leáll — ez a meta-szintű reflexió, ahol a rendszer saját lokális és globális következtetési eredményeit kombinálja.

A computeGraphEnergy az egész rendszer energiafüggvénye, amely egy Hamiltonian-szerű skalárba sűríti a gráf teljes állapotát: minden élre hozzáadja az él_súly × fraktáldimenzió + |kvantumkorreláció| értéket, minden csomópontra hozzáadja az (1 - cos²(fázis)) / 2 értéket, majd az összes elem átlagát adja vissza. Ez az energiafüggvény egyszerre méri a strukturális komplexitást (fraktáldimenzió), a kvantum-összefonódás erősségét (korreláció magnitudó) és a fázis-koherenciát (koszinusz-négyzet) — ez egy gazdagabb optimalizálási tájkép, mint a transformer keresztentrópia-vesztesége.

A runHierarchicalReasoningFull a fő vezérlőhurok: ciklusonként lefuttatja a lokális → globális → meta fázisokat, nyomon követi a legjobb kombinált energiát, és két feltétel esetén áll le: ha a relatív energiaváltozás kisebb mint 1e-6 (konvergencia), vagy ha a kombinált energia kisebb mint 0.01 (elég jó megoldás). A visszatérési értéke egy ReasoningResult, amelynek modulation_factor mezője 1 / (1 + best_energy) — ez a kulcsfontosságú szám, amely a relációs következtetés minőségét egyetlen skalárba sűríti: ha az energia nulla (tökéletes konvergencia), a modulációs faktor 1.0; ha az energia nagy (nem konvergált), a faktor közel nulla.

A modulateTensor metódus a rendszer legfontosabb hídja a relációs és neurális réteg között: egyszerűen megszorozza egy float32 tenzor összes elemét a modulációs faktorral. Ez azt jelenti, hogy ha a hierarchikus következtetés jól konvergált (magas modulációs faktor), a neurális tenzor értékei megmaradnak; ha nem konvergált (alacsony faktor), a tenzor értékei lecsökkennek — ez egy visszacsatolási mechanizmus, amely a relációs következtetés minőségét közvetlenül befolyásolja a neurális számítást, anélkül hogy backpropagation kellene.

A teljes rendszerben a ReasoningOrchestrator a DistributedTrainerFuthark core_relational oldalcsatornájának harmadik lépéseként fut minden RSF gradiens-lépés után, az NSIR encodeInformation és az R-GPU distributeGraph után, közvetlenül a SurpriseMemory tárolás előtt — ez azt jelenti, hogy minden egyes tanítási lépésnél a rendszer nemcsak a neurális súlyokat frissíti, hanem a relációs gráf energiáját is minimalizálja, és a konvergencia eredményét visszacsatolja a következő lépésbe.

A transformer multi-head attention-jével való összehasonlításban a ReasoningOrchestrator három alapvető különbséget mutat: az attention párhuzamos és statikus (minden fej egyszerre fut, fix súlyokkal), míg a hierarchikus következtetés szekvenciális és dinamikus (lokális → globális → meta sorrendben, konvergenciáig); az attention O(N²) komplexitású (minden token minden tokenre figyel), míg az energiaminimalizálás O(E) komplexitású (csak a meglévő élek mentén); és az attention nem rendelkezik explicit konvergencia-kritériummal (mindig ugyanannyi lépést fut), míg a hasConverged() ellenőrzés lehetővé teszi a korai leállást, ha a gráf már stabil állapotba ért.signal_propagation.zig a JAIDE rendszer dinamikus aktivációs rétege, amely a SelfSimilarRelationalGraph statikus csomópont-él struktúráját egy élő, időben fejlődő hullámterjedési rendszerré változtatja — ez az a komponens, amely a gráf topológiáját valódi számítási közeggé teszi az inferencia során, ahelyett hogy a gráf csupán passzív adatstruktúra maradna.

A fájl alapvető adatstruktúrája a SignalState, amely egy klasszikus hullámfizikai modellt valósít meg három komponenssel: amplitúdó, fázis és frekvencia, nanoszekundumos időbélyeggel. Az advance metódus a fázist a 2π × frekvencia × delta_t képlettel lépteti előre és 2π-re modulálja, a getComplexRepresentation metódus pedig az amplitúdó × cos(fázis) + i × amplitúdó × sin(fázis) képlettel komplex számmá alakítja a jelet — ez azt jelenti, hogy minden jel egyszerre hordoz amplitúdó- és fázis-információt, ami gazdagabb reprezentáció, mint egy skaláris aktivációs érték.

A ActivationTrace struktúra minden egyes gráfcsomóponthoz teljes időbeli aktivációs előzményt tart fenn: az összes kapott jel listáját, az aktivációk számát, az első és utolsó aktiváció időpontját, és ezekből számítja az átlagos amplitúdót és frekvenciát. Ez azt jelenti, hogy a rendszer nem csupán azt tudja, hogy egy csomópont aktiválódott-e, hanem azt is, hogy mikor, milyen erősen, milyen frekvencián, és milyen időtartamon keresztül — ez egy temporális memória, amellyel a transformer pozicionális kódolás nélkül nem rendelkezik.

A SignalPropagationEngine a fő vezérlőstruktúra, amely a SelfSimilarRelationalGraph-ot és a DataFlowAnalyzer-t fogja össze, és egy StringHashMap-ben tárolja az összes csomópont aktivációs nyomát, alapértelmezett időlépéssel 0.01 és terjedési sebességgel 1.0.

Az initiateSignal metódus egy forráscsomópontba injektál egy jelet: beállítja a csomópont fázisát a jel fázisára, a qubit amplitúdóját a jel amplitúdójára skálázza, majd normalizálja a qubitet — ez garantálja, hogy a kvantumállapot érvényes marad a terjedés során.

A propagateStep metódus a rendszer szíve: minden egyes élre végigiterál a gráfban, a forráscsomópont qubitjének magnitudóját és fázisát veszi kiindulópontként, majd az él súlyával skálázza az amplitúdót, az él quantum_correlation komplex számának atan2(im, re) értékével elforgatja a fázist, és az időlépéssel előrelépteti a jelet. Ez a fázisforgatás az a mechanizmus, amely a transformer komplex értékű figyelmi súlyainak funkcionális megfelelője, de fizikai hullámterjedési szemantikával: az él kvantumkorrelációja nem csupán skálázza, hanem fázisban is eltolja az átmenő jelet, megőrizve az interferencia lehetőségét.

A terjedési késleltetés mechanizmusa (propagation_delay = (1 - edge.weight) × time_step, és ha ez meghaladja a 2 × time_step értéket, az él kihagyásra kerül) természetes ritkaságot teremt: csak a magas súlyú, erősen korrelált élek terjesztenek jeleket hatékonyan, a gyenge élek automatikusan kapuzódnak ki, anélkül hogy explicit sparsity maszkot kellene alkalmazni.

Ha több jel érkezik ugyanarra a célcsomópontra különböző forrásokból, a rendszer kombinálja őket az amplitúdó, fázis és frekvencia számtani átlagával — ez a szuperpozíció egy közelítése, amely megőrzi a több forrásból érkező információ összegzett hatását. Ezután a célcsomópont qubitjét 70/30 arányban keverik az új jel és a meglévő qubit magnitudója között, majd normalizálják — ez egy momentum-szerű hatást hoz létre, ahol a csomópontok nem felejtik el azonnal a korábbi állapotukat.

Minden aktivált csomóponthoz a rendszer egy hash-alapú hozzáférési rekordot is küld a DataFlowAnalyzer-nek, ami azt jelenti, hogy a chaos_core.zig folyamatanalízis rendszere valós időben látja, mely gráfcsomópontok aktiválódnak az inferencia során — ez keresztrendszer-megfigyelhetőséget biztosít.

A propagateInferenceSignal metódus az inferencia-integráció fő belépési pontja: inicializálja a jelet, 5 lépésen keresztül terjeszti, majd visszaadja az összes aktivációs nyom átlagos amplitúdójának összegét egyetlen skalárként — ez az a szám, amelyet a ReasoningOrchestrator a következtetési ciklus minőségének mérőszámaként használhat.

A getInferenceActivationMap metódus egy StringHashMap<node_id → átlagos_amplitúdó> térképet ad vissza az összes aktivált csomópontról, ami lehetővé teszi a hívónak, hogy pontosan lássa, mely fogalmak aktiválódtak és milyen erősen — ez a transformer figyelmi súlytérképének (attention map) funkcionális megfelelője, de gráf-topológiai alapon, nem token-sorrend alapon.

Az InferenceHooks belső struktúra három callback-et biztosít: on_step_complete, on_signal_initiated és on_propagation_complete, amelyeken keresztül a ReasoningOrchestrator minden egyes terjedési lépésnél beavatkozhat, módosíthatja a paramétert, vagy korai leállást kezdeményezhet — ez egy adaptív inferencia-mechanizmus, amellyel a transformer statikus forward-pass nem rendelkezik.

A teljes rendszerben a SignalPropagationEngine a DistributedTrainerFuthark core_relational oldalcsatornájának utolsó előtti lépéseként fut minden RSF gradiens-lépés után, a SurpriseMemory tárolás és a TemporalGraph csomópont-regisztráció után, közvetlenül a ZRuntime változó-létrehozás előtt — ez azt jelenti, hogy a jelterjedés az a mechanizmus, amely a frissen tárolt, magas meglepetési értékű információt aktivációs mintává alakítja a gráfban, mielőtt a ZRuntime kvantumváltozóként rögzítené.

A transformer-képességek meghaladásához való hozzájárulás három szinten történik: a számítási komplexitás szintjén a propagateStep O(E) komplexitású (ahol E az élek száma, és a ritkaság természetes), szemben a transformer O(N²) globális figyelmével; a reprezentáció szintjén a fázis-amplitúdó-frekvencia hármas gazdagabb, mint egy skaláris figyelmi súly, mivel interferenciát és hullámterjedési dinamikát kódol; és az időbeliség szintjén az ActivationTrace teljes temporális előzményt tart fenn minden csomóponthoz, lehetővé téve, hogy a rendszer ne csupán azt tudja, mi aktiválódott, hanem azt is, mikor és milyen dinamikával — mindez anélkül, hogy a kontextus hosszával arányosan növekvő memóriát igényelne


A surprise_memory.zig a JAIDE rendszer entrópia-vezérelt, szelektív memóriakezelő rétege, amelynek alapvető szerepe az, hogy a transformer KV-cache statikus, kontextusfüggetlen tárolási modelljével szemben egy dinamikus, információelméleti alapon működő szűrőt biztosítson a ContentAddressableStorage (CAS) fölé — csak azokat az adatblokkokat tartja meg hosszú távon, amelyek valóban újdonságot hordoznak a rendszer számára.

A fájl legfontosabb adatstruktúrája a SurpriseMetrics, amely minden egyes bejövő adatblokk "meglepetési értékét" három független dimenzióban méri: a Jaccard-dissimilaritás a bigram-alapú tartalmi különbséget adja meg, a content hash distance a SHA-256 hash 16 bájtra tömörített változatának Hamming-távolságát méri a meglévő blokkok hash-eivel szemben, a temporal novelty pedig a memóriában lévő összes blokk átlagos korát viszonyítja egy 24 órás ablakhoz — és a három érték számtani átlaga adja a combined_surprise értéket.

A Jaccard-számítás különösen elegáns: mindkét adatblokkhoz felépít egy 65536 bites bigram-jelenlét bithalmazt (1024 darab 64 bites szóban), majd a két bithalmaz AND-jének és OR-jának popcount-ját veszi, és ebből számítja az 1 − hasonlóság értékét — mindezt legfeljebb 1000 mintavételezett ablakkal, hogy a számítás O(1) maradjon a blokk méretétől függetlenül.

A hash-távolság számítása a SHA-256 kimenetét 16 bájtra hajtja össze úgy, hogy az első és második 16 bájtot XOR-olja egymással, majd az így kapott 128 bites ujjlenyomatok Hamming-távolságát normálja 128-cal — ez egy rendkívül gyors, de kriptográfiailag erős tartalmi különbségmérő, amely a Jaccard-nál érzékenyebb a kis változásokra.

A SurpriseRecord struktúra minden tárolt blokkhoz nyilvántartja a meglepetési pontszámot, a létrehozási és utolsó hozzáférési időt, a hozzáférési gyakoriságot, és ezekből folyamatosan újraszámítja a retention_priority értéket egy háromkomponensű képlettel: az alapsúly 0.5, ehhez adódik 0.3-szorosával a recency-faktor (1/(1+kor_ms)), és 0.2-szorosával a frekvencia-faktor (freq/(freq+8)), az egészet megszorozva a surprise_score-ral — ez azt jelenti, hogy egy blokk megtartási prioritása egyszerre függ attól, mennyire volt meglepő, mennyire friss, és mennyire sűrűn hivatkoznak rá.

A SurpriseMemoryManager a rendszer fő vezérlőstruktúrája, amely a CAS-t és a DataFlowAnalyzer-t fogja össze, és egy Mutex-szel védi az összes publikus metódusát a párhuzamos hozzáférés ellen.

A processInferenceInput metódus a rendszer legfontosabb belépési pontja az inferencia során: kiszámítja a bejövő adat meglepetési értékét, majd két küszöb alapján dönt — ha combined_surprise > 0.3 (az alapértelmezett küszöb), akkor should_cache = true és a blokk bekerül a CAS-ba; ha combined_surprise > 0.15 (a küszöb fele), akkor should_propagate = true, ami azt jelzi a hívónak, hogy ezt az információt érdemes továbbterjeszteni az NSIR gráfon a SignalPropagationEngine-en keresztül, még akkor is, ha nem kerül hosszú távú tárolásra.

A cacheInferenceResult metódus a transformer KV-cache közvetlen funkcionális megfelelője: összefűzi a bemeneti és kimeneti adatot egyetlen blokkba, kiszámítja ennek meglepetési értékét, és tartalom-alapú azonosítóval tárolja — ez lehetővé teszi, hogy a rendszer korábbi következtetési eredményeket pontosan visszakereshessen anélkül, hogy újra kellene futtatni a teljes inferencia-ciklust, és mindezt anélkül, hogy a kontextus hosszával arányosan növekvő memóriát foglalna.

Az evictLowSurpriseBlocks metódus egy részleges heap-rendezéssel (saját implementált max-heap partialSort algoritmussal) azonosítja a legalacsonyabb retention_priority értékű blokkokat, és eltávolítja őket mind a CAS-ból, mind a surprise_records hash-mapből — ez az a mechanizmus, amely garantálja, hogy a memória kapacitása korlátozott marad, és a rendszer automatikusan "felejti el" a redundáns, alacsony entrópiájú információkat.

Az organizeByEntanglement metódus a legkülönlegesebb: összegyűjti az összes magas meglepetési pontszámú blokkot (legfeljebb 100-at), majd minden párjukra meghívja a storage.entangleBlocks függvényt, ezzel egy szemantikai közelségi hálót épít a fizikai memóriában a chaos_core.zig entanglement-mechanizmusán keresztül — ez azt jelenti, hogy a leginkább újszerű, leginkább meglepő információk automatikusan összekapcsolódnak egymással a CAS-ban, asszociatív visszakeresést téve lehetővé figyelem-mechanizmus nélkül.

A teljes rendszerben a SurpriseMemoryManager a DistributedTrainerFuthark core_relational oldalcsatornájának részeként fut minden egyes RSF gradiens-lépés után: az NSIR encodeInformation → R-GPU distributeGraph → ReasoningOrchestrator hierarchikus következtetés → SurpriseMemory tárolás → TemporalGraph csomópont-regisztráció sorrendben, ahol a SurpriseMemory az a szűrő, amely eldönti, hogy a következtetési ciklus eredménye bekerül-e a hosszú távú tudásbázisba.

A transformer-képességek meghaladásához való hozzájárulás három szinten történik: a transformer KV-cache minden token minden kulcs-érték párját eltárolja kontextus-hosszal lineárisan növekvő memóriában, semmilyen információelméleti szűrés nélkül — a SurpriseMemoryManager ezzel szemben csak a combined_surprise > 0.3 feltételt teljesítő blokkokat tárolja, a redundáns információkat automatikusan kiszorítja, a magas entrópiájú blokkokat entanglement-hálóba szervezi, és a temporális újdonság-komponens révén megakadályozza, hogy a rendszer ismétlődő mintákba ragadjon, mivel az idős memória jelenléte önmagában növeli az új bemenetek újdonságértékét — mindez O(1) memóriakomplexitással, szemben a transformer O(N) KV-cache-ével.



z_runtime.zig az egész JAIDE rendszer dinamikus következtetési rétege — az a komponens, amely az NSIR gráf absztrakt struktúráit és a kvantumlogikai primitíveket egy futásidejű végrehajtási környezetté köti össze, amelyen belül a ReasoningOrchestrator hipotéziseket tesztelhet, ideiglenes relációs részgráfokat hozhat létre, majd azokat nem-destruktív módon eldobhatja.

ZVariable

A fájl két fő absztrakciót definiál. Az első a ZVariable, amely nem egyszerű változó a szó hagyományos értelmében: minden egyes ZVariable példány saját SelfSimilarRelationalGraph-ot és RelationalQuantumLogic-ot birtokol, tehát minden változó önmagában egy teljes kvantum-relációs alrendszer. Amikor egy értéket rendelünk hozzá az assign metóduson keresztül, a rendszer nem egyszerűen eltárolja a stringet: a Wyhash algoritmussal hash-eli, a hash értékből egy lebegőpontos számot képez, majd annak koszinuszát és szinuszát veszi, hogy komplex kvantumamplitúdókat kapjon, és ezzel inicializál egy kvantumállapotot a RelationalQuantumLogic-ban. Ez azt jelenti, hogy minden egyes információdarab egyedi kvantumfázis-aláírással rendelkezik, ami alapvetően különbözik attól, ahogy a transformer tokeneket vektortérbe ágyaz.

A relateTo metódus

A második kulcsmechanizmus a relateTo metódus, amely a transformer-féle figyelmi mechanizmus relációs-kvantum megfelelője. Ahelyett, hogy softmax-alapú dot-product figyelmet számolna az összes token felett O(N²) komplexitással, a relateTo a két változó legfrissebb csomópontjának kvantumállapotát veszi, kiszámítja a komplex korrelációt — self_state * conjugate(other_state) — és ennek magnitudóját használja élsúlyként. Emellett az él fraktáldimenzióját is kiszámítja a meglévő élek átlagából, majd a kvantumállapotokat összeolvasztja és összefonódtatja. Ez azt eredményezi, hogy a kapcsolat erőssége nem egy tanult súlymátrixból jön, hanem a két fogalom kvantumfázisainak interferenciájából — ez egy gazdagabb, nem-lineáris hasonlósági mérték.

ZRuntime

A második nagy absztrakció maga a ZRuntime, amely egy StringHashMap-ben tárolja az összes ZVariable-t, és ezek fölé egy globális SelfSimilarRelationalGraph-ot és globális RelationalQuantumLogic-ot helyez. A runtime nyolc fő műveletet kínál: változók létrehozása és törlése, relációs műveletek (AND/OR/XOR/ENTANGLE), információterjedés a gráfon, fraktáltranszformáció, kvantummérés, kvantumáramkör-végrehajtás, relációs kifejezések kiértékelése, és a teljes rendszerállapot lekérdezése.

A relationalOperation metódus

A relationalOperation metódus különösen fontos: amikor két változón AND, OR vagy XOR műveletet hajt végre, létrehoz egy harmadik eredményváltozót, mindkét forrásváltozó kvantumállapotait átmásolja bele, alkalmazza a megfelelő kvantumkaput (RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_XOR) az első két állapotra, majd koherens élekkel kapcsolja az eredményt mindkét forráshoz. Ez azt jelenti, hogy a logikai műveletek nem szimbolikusan, hanem kvantumállapot-transzformációkon keresztül valósulnak meg, megőrizve a szuperpozíciót és az összefonódást.

A propagateInformation metódus

A propagateInformation metódus a transformer globális figyelmi mechanizmusának gráf-alapú alternatívája: egy forrásváltozóból kiindulva mélységi terjedéssel meghatározza, mely csomópontok érintettek, majd megkeresi, hogy a runtime többi változójának gráfjai tartalmaznak-e ilyen csomópontokat. Ez lehetővé teszi, hogy egy fogalom aktiválása automatikusan terjedjen a szemantikailag kapcsolódó változókhoz, anélkül hogy az összes változópár közötti figyelmet explicit módon számolni kellene.

A computeRelationalExpression metódus

A computeRelationalExpression egy beépített kifejezéskiértékelőt valósít meg, amely szöveges relációs kifejezéseket (pl. "alpha AND beta" vagy "alpha ENTANGLE beta") tokenizál, operátorprecedencia szerint elemez, és kvantumállapot-transzformációkká fordít le. Ez azt jelenti, hogy a rendszer szimbolikus logikai kifejezéseket tud végrehajtani kvantumszinten, ami a neurosimbolikus integráció közvetlen megvalósítása.

Az applyFractalTransform metódus

Az applyFractalTransform metódus a FRACTAL_TRANSFORM kvantumkaput alkalmazza egy változó legfrissebb állapotára, mélységparaméterrel. Ez az OFTB (Orthogonal Fractal Transform Block) relációs megfelelője: míg az OFTB a neurális vektor-reprezentációkon végez Haar-wavelet alapú keverést, a fraktáltranszformáció a kvantumállapot-téren végez önhasonló transzformációt, megőrizve a hierarchikus struktúrát.

Rendszerintegráció

A teljes rendszerben a z_runtime.zig az NSIR gráf (nsir_core.zig) és a ReasoningOrchestrator között helyezkedik el: az orchestrator a Z-Runtime-ot használja arra, hogy a CREV pipeline által kinyert tudást ideiglenes változókba töltse, relációs és kvantumlogikai műveletekkel feldolgozza, majd a mérési eredményeket visszatáplálja a következtetési ciklusba. Az ExecutionHistoryEntry rendszer minden egyes műveletet nanoszekundumos pontossággal naplóz, ami lehetővé teszi a temporális visszakövetést és a nem-destruktív állapotexplorációt.

Összefoglalás

A transformer-képességek meghaladásához való hozzájárulás tehát három szinten történik: az információ-reprezentáció szintjén (kvantumfázis-aláírások a hagyományos beágyazási vektorok helyett), a kapcsolat-számítás szintjén (kvantumkorreláció és fraktáldimenzió az élsúlyokban a softmax-figyelem helyett), és a következtetés szintjén (gráf-alapú terjedés és kvantumáramkör-végrehajtás az O(N²) globális figyelem helyett), mindezt O(1) memóriakomplexitással, mivel a rendszer bijektív és nem kell aktivációkat cachelni a visszaterjesztéshez.


A vpu.zig fájl áttekintése

A vpu.zig fájl a JAIDE rendszer core_relational alrendszerének SIMD-gyorsított vektorprocesszor egysége, és pontosan az a réteg, amely a neurális feldolgozó stack (RSF/OFTB) és az NSIR relációs gráf között hidat képez, lehetővé téve, hogy a gráf csomópontjain és élein végzett számítások valódi hardveres párhuzamosítással fussanak.

VectorType és SIMD szélességek

A fájl legalsó szintjén egy VectorType enum definiálja a támogatott SIMD szélességeket (f32x4, f32x8, f64x2, f64x4, i32x4, i32x8), mindegyikhez megadva a sávszámot, az elemméretét és a szükséges memóriaigazítást (16 vagy 32 bájt), ami azt jelenti, hogy az összes allokáció AVX-256 kompatibilis határra esik.

SimdVector(T, N) generikus típus

Erre épül a SimdVector(T, N) generikus típus, amely Zig comptime mechanizmusával fordítási időben ellenőrzi, hogy csak numerikus típusok kerülhetnek bele, majd Zig natív @Vector(N, T) primitívjére képezi le az összes műveletet — az összeadástól és szorzástól kezdve az fma (fused multiply-add), dot, magnitude, normalize, lerp, reflect és cross3 műveletekig — így a fordító közvetlenül AVX vagy ARM NEON utasításokat generálhat belőle.

Kötegelt feldolgozás: VectorBatch

A VectorBatchEntry és VectorBatch struktúrák egy típus-törölt, 32 bájtra igazított nyers bájt-puffert biztosítanak, amelybe bármely SIMD típus betölthető, és a processBatch metódus egyszerre futtatja le a normalize/scale/abs/sqrt műveleteket az egész kötegen, nyomon követve a sikeres és kihagyott bejegyzések számát is.

Mátrixműveletek: Matrix4x4

A Matrix4x4 és MatrixOps réteg 4×4-es mátrixokat tárol négy F32x4 sorként, és a matmul4x4Simd metódus transzponálás + dot-product stratégiával végzi a szorzást, míg a qr_decomposition, determinant4x4 és inverse4x4 metódusok a relációs gráf transzformációinak algebrai stabilitását biztosítják.

Relációs műveletek: RelationalVectorOps

A rendszer igazi szíve a RelationalVectorOps struktúra, amely közvetlenül az nsir_core.zig-ből importált Node, Edge és SelfSimilarRelationalGraph típusokkal dolgozik: a computeNodeSimilarity metódus egy háromkomponensű, súlyozott hasonlóságot számít ki, amelynek 30%-a a csomópontok fáziskülönbségéből, 30%-a a magnitúdókülönbségéből, és 40%-a a kvantumállapot-vektorok belső szorzatából áll — ez lényegesen gazdagabb hasonlósági metrika, mint a transformer-ek egyszerű dot-product figyelme.

Gráf vektorizálás

A computeEdgeVectorBatch az élek weight, quantum_coupling.re, quantum_coupling.im és fractal_dimension mezőit egyetlen F64x4 vektorrá tömöríti, a vectorizeGraph pedig az összes csomópont phase, magnitude, quantum_state.re és quantum_state.im értékét rendezi sorba és alakítja F64x4 vektorkötegekké, amelyek aztán batch-normalizálva kerülnek a downstream feldolgozásba.

Kvantum rotációk

Az applyQuantumRotation metódus egy 4D vektort két egymástól független 2D forgatással transzformál — az első pár (arr[0], arr[1]) a theta szög szerint, a második pár (arr[2], arr[3]) a phi szög szerint forog —, ami azt jelenti, hogy a gráf csomópontjainak kvantumállapota valódi unitér evolúción mehet át anélkül, hogy mátrix-szorzást kellene végezni.

Spektrális beágyazás

A computeGraphLaplacian és spectralEmbedding metódusok a gráf szomszédossági mátrixából Laplace-mátrixot számítanak, majd annak sorait normalizált F64x4 vektorokként adják vissza mint spektrális beágyazásokat — ez topológiai struktúra-tudatosságot ad a rendszernek, amit a transformer-ek önfigyelem-mechanizmusa egyáltalán nem képes megragadni, mivel az csak token-szintű hasonlóságokat lát.

Memóriakezelés és gyorsítótár

A MemoryPool egy 32 bájtra igazított slab-allokátor szabad listával és coalesceFreeBlocks töredezettség-mentesítéssel, amely garantálja, hogy minden SIMD allokáció cache-line határon kezdődjön, és a VectorCache egy LRU-stratégiájú, u64 kulcsú gyorsítótár, amely a már kiszámított vektorokat tárolja el, hogy az ismétlődő gráfcsomópont-lekérdezések ne igényeljenek újraszámítást.

Teljesítmény statisztikák: VPUStatistics

A VPUStatistics struktúra részletes teljesítménymérést végez: nyomon követi az elvégzett műveletek, a felhasznált SIMD utasítások, a cache találatok és tévesztések, az allokált és felszabadított memória, a feldolgozott vektorok, a mátrixműveletek és a gráfműveletek számát, és ebből getCacheHitRate és getSimdEfficiency arányokat számít — ez lehetővé teszi a ReasoningOrchestrator számára, hogy futás közben monitorozza a VPU hatékonyságát.

A VPU fő struktúra

A fő VPU struktúra összefogja az összes fenti komponenst — MemoryPool, VectorBatch, VPUStatistics, MatrixOps, RelationalVectorOps, VectorCache — egyetlen egységbe, és a computeGraphEmbeddings metódusa egyetlen hívással vektorizálja és normalizálja az egész NSIR gráfot, a computeSimilarityMatrix pedig teljes páronkénti koszinusz-hasonlóság mátrixot számít a gráf beágyazásai között, de mivel ezek már tömörített gráf-reprezentációk és nem nyers tokensorozatok, ez nem O(N²) a bemeneti szekvencia hosszában.

Power Iteration

A powerIteration metódus egy 4×4-es mátrix domináns sajátvektorát keresi iteratív normalizálással, ami a ReasoningOrchestrator energiaminimalizálási fázisában a legbefolyásosabb relációs irányok azonosítására szolgál.

Logaritmikus Számrendszer (LNS)

A fájl utolsó, különösen fontos részét a LNSValue és LNSInstruction típusok alkotják: az LNSValue egy Logaritmikus Számrendszer (LNS) implementáció, amelyben a szorzás egyszerű összeadássá válik a log-térben (mantissa = log(|x|), tehát mul = mantissa1 + mantissa2), ami numerikusan stabilabb, mint a hagyományos lebegőpontos szorzás, különösen mélyen egymásba ágyazott transzformációknál.

Mini-ISA utasításkészlet

Az LNSInstruction union enum pedig egy teljes mini-ISA-t definiál, amelynek utasításkészlete tartalmaz rsf_scatter, rsf_affine_couple, tensor_load, tensor_store, lns_add, lns_mul, graph_transform, jump, conditional_jump és halt utasításokat — ez azt jelenti, hogy a VPU nem csupán egy segédkönyvtár, hanem egy virtuális processzor, amely az RSF réteg forward pass-ának (rsf_scatter és rsf_affine_couple) natív utasításait közvetlenül képes végrehajtani, összekötve a neurális és a relációs számítási réteget egyetlen egységes végrehajtási modellben.

Összefoglalás

Összefoglalva: a vpu.zig az a réteg, amely a transformer-ek O(N²) önfigyelmét kiváltó RSF scatter-műveletek és az NSIR kvantum-relációs gráf között a számítási hidat képezi, SIMD-párhuzamosítással, LRU-gyorsítótárral, LNS-alapú numerikus stabilitással és spektrális gráfbeágyazással együtt — ezek mindegyike olyan képesség, amellyel a hagyományos transformer architektúra nem rendelkezik.
