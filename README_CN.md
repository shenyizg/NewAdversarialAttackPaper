# Latest Adversarial Attack Papers
**update at 2026-01-04 08:58:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Towards Provably Secure Generative AI: Reliable Consensus Sampling**

迈向可证明安全的生成式AI：可靠的共识抽样 cs.CR

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24925v1) [paper-pdf](https://arxiv.org/pdf/2512.24925v1)

**Authors**: Yu Cui, Hang Fu, Sicheng Pan, Zhuoyu Sun, Yifei Liu, Yuhong Nie, Bo Ran, Baohan Huang, Xufeng Zhang, Haibin Zhang, Cong Zuo, Licheng Wang

**Abstract**: Existing research on generative AI security is primarily driven by mutually reinforcing attack and defense methodologies grounded in empirical experience. This dynamic frequently gives rise to previously unknown attacks that can circumvent current detection and prevention. This necessitates the continual updating of security mechanisms. Constructing generative AI with provable security and theoretically controllable risk is therefore necessary. Consensus Sampling (CS) is a promising algorithm toward provably secure AI. It controls risk by leveraging overlap in model output probabilities. However, we find that CS relies on frequent abstention to avoid unsafe outputs, which reduces utility. Moreover, CS becomes highly vulnerable when unsafe models are maliciously manipulated. To address these issues, we propose a new primitive called Reliable Consensus Sampling (RCS), that traces acceptance probability to tolerate extreme adversarial behaviors, improving robustness. RCS also eliminates the need for abstention entirely. We further develop a feedback algorithm to continuously and dynamically enhance the safety of RCS. We provide theoretical guarantees that RCS maintains a controllable risk threshold. Extensive experiments show that RCS significantly improves robustness and utility while maintaining latency comparable to CS. We hope this work contributes to the development of provably secure generative AI.

摘要: 现有的生成性人工智能安全研究主要由基于经验的相互强化的攻击和防御方法驱动。这种动态经常引发以前未知的攻击，这些攻击可以规避当前的检测和预防。这需要不断更新安全机制。因此，构建具有可证明的安全性和理论上可控风险的生成性人工智能是必要的。一致性抽样（CS）是一种很有前途的可证明安全的人工智能算法。它通过利用模型输出概率中的重叠来控制风险。然而，我们发现，CS依赖于频繁的预防，以避免不安全的输出，这降低了效用。此外，当不安全的模型被恶意操纵时，CS变得非常脆弱。为了解决这些问题，我们提出了一个新的原语，称为可靠共识采样（RCS），跟踪接受概率，以容忍极端的对抗行为，提高鲁棒性。RCS还完全消除了对预防的需要。我们进一步开发了一个反馈算法，以持续和动态地提高RCS的安全性。我们提供理论保证，RCS保持一个可控的风险阈值。大量实验表明，RC显着提高了稳健性和实用性，同时保持了与CS相当的延迟。我们希望这项工作有助于发展可证明安全的生成性人工智能。



## **2. Projection-based Adversarial Attack using Physics-in-the-Loop Optimization for Monocular Depth Estimation**

使用物理在环优化进行单目深度估计的基于投影的对抗攻击 cs.CV

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24792v1) [paper-pdf](https://arxiv.org/pdf/2512.24792v1)

**Authors**: Takeru Kusakabe, Yudai Hirose, Mashiho Mukaida, Satoshi Ono

**Abstract**: Deep neural networks (DNNs) remain vulnerable to adversarial attacks that cause misclassification when specific perturbations are added to input images. This vulnerability also threatens the reliability of DNN-based monocular depth estimation (MDE) models, making robustness enhancement a critical need in practical applications. To validate the vulnerability of DNN-based MDE models, this study proposes a projection-based adversarial attack method that projects perturbation light onto a target object. The proposed method employs physics-in-the-loop (PITL) optimization -- evaluating candidate solutions in actual environments to account for device specifications and disturbances -- and utilizes a distributed covariance matrix adaptation evolution strategy. Experiments confirmed that the proposed method successfully created adversarial examples that lead to depth misestimations, resulting in parts of objects disappearing from the target scene.

摘要: 深度神经网络（DNN）仍然容易受到对抗攻击，当向输入图像添加特定扰动时，这些攻击会导致错误分类。该漏洞还威胁到基于DNN的单目深度估计（MDE）模型的可靠性，使得鲁棒性增强成为实际应用中的关键需求。为了验证基于DNN的MDE模型的脆弱性，本研究提出了一种基于投影的对抗攻击方法，将扰动光投影到目标对象上。所提出的方法采用物理在环（PITL）优化--评估实际环境中的候选解决方案以考虑设备规格和干扰--并利用分布式协方差矩阵自适应进化策略。实验证实，所提出的方法成功创建了导致深度误判的对抗性示例，导致部分对象从目标场景中消失。



## **3. HeteroHBA: A Generative Structure-Manipulating Backdoor Attack on Heterogeneous Graphs**

HeteroDBA：对异类图的生成性结构操纵后门攻击 cs.LG

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24665v1) [paper-pdf](https://arxiv.org/pdf/2512.24665v1)

**Authors**: Honglin Gao, Lan Zhao, Junhao Ren, Xiang Li, Gaoxi Xiao

**Abstract**: Heterogeneous graph neural networks (HGNNs) have achieved strong performance in many real-world applications, yet targeted backdoor poisoning on heterogeneous graphs remains less studied. We consider backdoor attacks for heterogeneous node classification, where an adversary injects a small set of trigger nodes and connections during training to force specific victim nodes to be misclassified into an attacker-chosen label at test time while preserving clean performance. We propose HeteroHBA, a generative backdoor framework that selects influential auxiliary neighbors for trigger attachment via saliency-based screening and synthesizes diverse trigger features and connection patterns to better match the local heterogeneous context. To improve stealthiness, we combine Adaptive Instance Normalization (AdaIN) with a Maximum Mean Discrepancy (MMD) loss to align the trigger feature distribution with benign statistics, thereby reducing detectability, and we optimize the attack with a bilevel objective that jointly promotes attack success and maintains clean accuracy. Experiments on multiple real-world heterogeneous graphs with representative HGNN architectures show that HeteroHBA consistently achieves higher attack success than prior backdoor baselines with comparable or smaller impact on clean accuracy; moreover, the attack remains effective under our heterogeneity-aware structural defense, CSD. These results highlight practical backdoor risks in heterogeneous graph learning and motivate the development of stronger defenses.

摘要: 异类图神经网络（HGNN）在许多现实世界的应用中取得了强劲的性能，但对异类图的有针对性的后门中毒研究仍然较少。我们考虑了针对异类节点分类的后门攻击，其中对手在训练期间注入一小组触发节点和连接，以迫使特定的受害者节点在测试时被错误分类到攻击者选择的标签中，同时保持干净的性能。我们提出了HeteroDBA，这是一种生成式后门框架，通过基于显著性的筛选来选择有影响力的辅助邻居进行触发附件，并综合各种触发特征和连接模式，以更好地匹配本地的异类上下文。为了提高隐蔽性，我们将自适应实例规范化（AdaIN）与最大均值偏差（MMD）损失相结合，以使触发特征分布与良性统计数据保持一致，从而降低可检测性，并且我们通过两层目标优化攻击，共同促进攻击成功并保持清晰的准确性。在具有代表性HGNN架构的多个现实世界的异类图上进行的实验表明，HeteroDBA始终比之前的后门基线获得更高的攻击成功，对清理准确性的影响相当或更小;此外，在我们的异类感知结构防御下，攻击仍然有效。这些结果凸显了异类图学习中的实际后门风险，并激励开发更强大的防御。



## **4. CPR: Causal Physiological Representation Learning for Robust ECG Analysis under Distribution Shifts**

CPR：用于分布偏移下的稳健ECG分析的因果生理表示学习 cs.LG

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24564v1) [paper-pdf](https://arxiv.org/pdf/2512.24564v1)

**Authors**: Shunbo Jia, Caizhi Liao

**Abstract**: Deep learning models for Electrocardiogram (ECG) diagnosis have achieved remarkable accuracy but exhibit fragility against adversarial perturbations, particularly Smooth Adversarial Perturbations (SAP) that mimic biological morphology. Existing defenses face a critical dilemma: Adversarial Training (AT) provides robustness but incurs a prohibitive computational burden, while certified methods like Randomized Smoothing (RS) introduce significant inference latency, rendering them impractical for real-time clinical monitoring. We posit that this vulnerability stems from the models' reliance on non-robust spurious correlations rather than invariant pathological features. To address this, we propose Causal Physiological Representation Learning (CPR). Unlike standard denoising approaches that operate without semantic constraints, CPR incorporates a Physiological Structural Prior within a causal disentanglement framework. By modeling ECG generation via a Structural Causal Model (SCM), CPR enforces a structural intervention that strictly separates invariant pathological morphology (P-QRS-T complex) from non-causal artifacts. Empirical results on PTB-XL demonstrate that CPR significantly outperforms standard clinical preprocessing methods. Specifically, under SAP attacks, CPR achieves an F1 score of 0.632, surpassing Median Smoothing (0.541 F1) by 9.1%. Crucially, CPR matches the certified robustness of Randomized Smoothing while maintaining single-pass inference efficiency, offering a superior trade-off between robustness, efficiency, and clinical interpretability.

摘要: 用于心电图（心电图）诊断的深度学习模型已经实现了显着的准确性，但对对抗性扰动，特别是模拟生物形态的平滑对抗性扰动（SAP）表现出脆弱性。现有的防御面临着一个关键的困境：对抗训练（AT）提供鲁棒性，但会带来令人望而却步的计算负担，而随机平滑（RS）等认证方法会引入显着的推理延迟，使其对于实时临床监测来说不切实际。我们推测，这种漏洞源于模型对非稳健虚假相关性的依赖，而不是不变的病理特征。为了解决这个问题，我们提出了因果生理表示学习（CPR）。与在没有语义约束的情况下运行的标准去噪方法不同，CPR将生理结构先验纳入因果分离框架中。通过通过结构性因果模型（SCP）对心电图生成进行建模，CPR实施了一种结构性干预，将不变的病理形态（P-SVR-T复合体）与非因果伪影严格分离。PTB-XL的经验结果表明，CPR显着优于标准临床预处理方法。具体来说，在SAP攻击下，CPR的F1评分为0.632，超过中位数平滑（0.541 F1）9.1%。至关重要的是，CPR与随机平滑的认证稳健性相匹配，同时保持单程推理效率，在稳健性、效率和临床可解释性之间提供了卓越的权衡。



## **5. Training-Free Color-Aware Adversarial Diffusion Sanitization for Diffusion Stegomalware Defense at Security Gateways**

安全门户处的扩散隐格软件防御免培训色彩感知对抗扩散消毒 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24499v1) [paper-pdf](https://arxiv.org/pdf/2512.24499v1)

**Authors**: Vladimir Frants, Sos Agaian

**Abstract**: The rapid expansion of generative AI has normalized large-scale synthetic media creation, enabling new forms of covert communication. Recent generative steganography methods, particularly those based on diffusion models, can embed high-capacity payloads without fine-tuning or auxiliary decoders, creating significant challenges for detection and remediation. Coverless diffusion-based techniques are difficult to counter because they generate image carriers directly from secret data, enabling attackers to deliver stegomalware for command-and-control, payload staging, and data exfiltration while bypassing detectors that rely on cover-stego discrepancies. This work introduces Adversarial Diffusion Sanitization (ADS), a training-free defense for security gateways that neutralizes hidden payloads rather than detecting them. ADS employs an off-the-shelf pretrained denoiser as a differentiable proxy for diffusion-based decoders and incorporates a color-aware, quaternion-coupled update rule to reduce artifacts under strict distortion limits. Under a practical threat model and in evaluation against the state-of-the-art diffusion steganography method Pulsar, ADS drives decoder success rates to near zero with minimal perceptual impact. Results demonstrate that ADS provides a favorable security-utility trade-off compared to standard content transformations, offering an effective mitigation strategy against diffusion-driven steganography.

摘要: 生成性人工智能的快速扩张使大规模合成媒体创建正常化，从而实现新形式的秘密传播。最近的生成式隐写术方法，特别是基于扩散模型的方法，可以嵌入高容量有效负载，而无需微调或辅助解码器，这给检测和修复带来了重大挑战。基于无掩护扩散的技术很难对抗，因为它们直接从秘密数据生成图像载体，使攻击者能够提供用于命令与控制、有效负载分级和数据泄露的隐写软件，同时绕过依赖于覆盖隐写差异的检测器。这项工作引入了对抗扩散消毒（ADS），这是一种针对安全网关的免训练防御，可以中和隐藏的有效负载而不是检测它们。ADS采用现成的预训练降噪器作为基于扩散的解码器的可区分代理，并结合颜色感知、四元数耦合更新规则，以减少严格失真限制下的伪影。在实际的威胁模型下，并在针对最先进的扩散隐写术方法Pulsar进行评估时，ADS将解码器成功率降至接近零，而对感知的影响最小。结果表明，与标准内容转换相比，ADS提供了有利的安全性与效用权衡，为针对扩散驱动的隐写术提供了有效的缓解策略。



## **6. RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation**

RAGPart & RAGMass：检索增强一代中的检索阶段防御体中毒 cs.IR

Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24268v1) [paper-pdf](https://arxiv.org/pdf/2512.24268v1)

**Authors**: Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-Liess, Cho-Yu Jason Chiang, Furong Huang

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to enhance large language models (LLMs) with external knowledge, reducing hallucinations and compensating for outdated information. However, recent studies have exposed a critical vulnerability in RAG pipelines corpus poisoning where adversaries inject malicious documents into the retrieval corpus to manipulate model outputs. In this work, we propose two complementary retrieval-stage defenses: RAGPart and RAGMask. Our defenses operate directly on the retriever, making them computationally lightweight and requiring no modification to the generation model. RAGPart leverages the inherent training dynamics of dense retrievers, exploiting document partitioning to mitigate the effect of poisoned points. In contrast, RAGMask identifies suspicious tokens based on significant similarity shifts under targeted token masking. Across two benchmarks, four poisoning strategies, and four state-of-the-art retrievers, our defenses consistently reduce attack success rates while preserving utility under benign conditions. We further introduce an interpretable attack to stress-test our defenses. Our findings highlight the potential and limitations of retrieval-stage defenses, providing practical insights for robust RAG deployments.

摘要: 检索增强生成（RAG）已成为一种有前途的范式，可以利用外部知识增强大型语言模型（LLM），减少幻觉并补偿过时信息。然而，最近的研究暴露了RAG管道库中毒中的一个关键漏洞，即对手将恶意文档注入检索库以操纵模型输出。在这项工作中，我们提出了两种补充的检索阶段防御：RAGPart和RAGMass。我们的防御系统直接在寻回犬上运行，使它们在计算上轻量级，并且不需要修改生成模型。RAGPart利用密集检索器的固有训练动态，利用文档分区来减轻中毒点的影响。相比之下，RAGMass根据目标令牌屏蔽下的显着相似性变化来识别可疑令牌。通过两种基准、四种中毒策略和四种最先进的寻回犬，我们的防御系统持续降低攻击成功率，同时在良性条件下保持实用性。我们进一步引入可解释的攻击来压力测试我们的防御。我们的研究结果强调了回收阶段防御的潜力和局限性，为稳健的RAG部署提供了实用的见解。



## **7. How Would Oblivious Memory Boost Graph Analytics on Trusted Processors?**

不经意记忆如何增强受信任处理器上的图形分析？ cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24255v1) [paper-pdf](https://arxiv.org/pdf/2512.24255v1)

**Authors**: Jiping Yu, Xiaowei Zhu, Kun Chen, Guanyu Feng, Yunyi Chen, Xiaoyu Fan, Wenguang Chen

**Abstract**: Trusted processors provide a way to perform joint computations while preserving data privacy. To overcome the performance degradation caused by data-oblivious algorithms to prevent information leakage, we explore the benefits of oblivious memory (OM) integrated in processors, to which the accesses are unobservable by adversaries. We focus on graph analytics, an important application vulnerable to access-pattern attacks. With a co-design between storage structure and algorithms, our prototype system is 100x faster than baselines given an OM sized around the per-core cache which can be implemented on existing processors with negligible overhead. This gives insights into equipping trusted processors with OM.

摘要: 受信任的处理器提供了一种在保护数据隐私的同时执行联合计算的方法。为了克服数据不经意算法造成的性能下降以防止信息泄露，我们探索了集成在处理器中的不经意存储器（OM）的好处，对手无法观察到对其的访问。我们专注于图形分析，这是一个容易受到访问模式攻击的重要应用程序。通过存储结构和算法之间的协同设计，我们的原型系统比基线快100倍，因为OM的大小围绕每核高速缓存，可以在现有处理器上实现，并且可以忽略不计的开销。这为如何为值得信赖的处理器配备OM提供了见解。



## **8. Guided Diffusion-based Generation of Adversarial Objects for Real-World Monocular Depth Estimation Attacks**

针对现实世界单目深度估计攻击的基于扩散的对抗对象的引导生成 cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24111v1) [paper-pdf](https://arxiv.org/pdf/2512.24111v1)

**Authors**: Yongtao Chen, Yanbo Wang, Wentao Zhao, Guole Shen, Tianchen Deng, Jingchuan Wang

**Abstract**: Monocular Depth Estimation (MDE) serves as a core perception module in autonomous driving systems, but it remains highly susceptible to adversarial attacks. Errors in depth estimation may propagate through downstream decision making and influence overall traffic safety. Existing physical attacks primarily rely on texture-based patches, which impose strict placement constraints and exhibit limited realism, thereby reducing their effectiveness in complex driving environments. To overcome these limitations, this work introduces a training-free generative adversarial attack framework that generates naturalistic, scene-consistent adversarial objects via a diffusion-based conditional generation process. The framework incorporates a Salient Region Selection module that identifies regions most influential to MDE and a Jacobian Vector Product Guidance mechanism that steers adversarial gradients toward update directions supported by the pre-trained diffusion model. This formulation enables the generation of physically plausible adversarial objects capable of inducing substantial adversarial depth shifts. Extensive digital and physical experiments demonstrate that our method significantly outperforms existing attacks in effectiveness, stealthiness, and physical deployability, underscoring its strong practical implications for autonomous driving safety assessment.

摘要: 单目深度估计（MDE）是自动驾驶系统中的核心感知模块，但它仍然极易受到对抗攻击。深度估计中的错误可能会通过下游决策传播并影响整体交通安全。现有的物理攻击主要依赖于基于纹理的补丁，这些补丁施加了严格的放置限制并表现出有限的真实感，从而降低了它们在复杂驾驶环境中的有效性。为了克服这些限制，这项工作引入了一个免训练的生成式对抗攻击框架，该框架通过基于扩散的条件生成过程生成自然主义、场景一致的对抗对象。该框架包含一个显著区域选择模块，用于识别对MDE最有影响的区域，以及一个雅可比向量乘积指导机制，用于将对抗梯度转向由预训练扩散模型支持的更新方向。这个公式使得能够产生物理上合理的对抗性对象，能够引起实质性的对抗性深度偏移。大量的数字和物理实验表明，我们的方法在有效性、隐蔽性和物理可部署性方面明显优于现有攻击，强调了其对自动驾驶安全评估的强大实际意义。



## **9. Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the LLM Safety Arms Race?**

越狱攻击与内容安全过滤器：我们在LLM安全军备竞赛中走了多远？ cs.CR

26 pages,11 tables, 7 figures

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24044v1) [paper-pdf](https://arxiv.org/pdf/2512.24044v1)

**Authors**: Yuan Xin, Dingfan Chen, Linyi Yang, Michael Backes, Xiao Zhang

**Abstract**: As large language models (LLMs) are increasingly deployed, ensuring their safe use is paramount. Jailbreaking, adversarial prompts that bypass model alignment to trigger harmful outputs, present significant risks, with existing studies reporting high success rates in evading common LLMs. However, previous evaluations have focused solely on the models, neglecting the full deployment pipeline, which typically incorporates additional safety mechanisms like content moderation filters. To address this gap, we present the first systematic evaluation of jailbreak attacks targeting LLM safety alignment, assessing their success across the full inference pipeline, including both input and output filtering stages. Our findings yield two key insights: first, nearly all evaluated jailbreak techniques can be detected by at least one safety filter, suggesting that prior assessments may have overestimated the practical success of these attacks; second, while safety filters are effective in detection, there remains room to better balance recall and precision to further optimize protection and user experience. We highlight critical gaps and call for further refinement of detection accuracy and usability in LLM safety systems.

摘要: 随着大型语言模型（LLM）的部署越来越多，确保它们的安全使用至关重要。越狱、对抗性促使绕过模型对齐引发有害输出，带来重大风险，现有研究报告称，规避常见LLM的成功率很高。然而，之前的评估仅关注模型，忽视了完整的部署管道，该管道通常包含内容审核过滤器等额外的安全机制。为了解决这一差距，我们首次对针对LLM安全调整的越狱攻击进行了系统评估，评估了它们在整个推理管道（包括输入和输出过滤阶段）中的成功。我们的研究结果得出了两个关键见解：首先，几乎所有评估的越狱技术都可以被至少一个安全过滤器检测到，这表明之前的评估可能高估了这些攻击的实际成功;其次，虽然安全过滤器在检测方面有效，但仍有空间更好地平衡召回和精确度，以进一步优化保护和用户体验。我们强调了关键差距，并呼吁进一步完善LLM安全系统的检测准确性和可用性。



## **10. RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress**

重复诅咒：测量和理解在OSS压力下混合专家LLM中的路由器不平衡 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23995v1) [paper-pdf](https://arxiv.org/pdf/2512.23995v1)

**Authors**: Ruixuan Huang, Qingyue Wang, Hantao Huang, Yudong Gao, Dong Chen, Shuai Wang, Wei Wang

**Abstract**: Mixture-of-Experts architectures have become the standard for scaling large language models due to their superior parameter efficiency. To accommodate the growing number of experts in practice, modern inference systems commonly adopt expert parallelism to distribute experts across devices. However, the absence of explicit load balancing constraints during inference allows adversarial inputs to trigger severe routing concentration. We demonstrate that out-of-distribution prompts can manipulate the routing strategy such that all tokens are consistently routed to the same set of top-$k$ experts, which creates computational bottlenecks on certain devices while forcing others to idle. This converts an efficiency mechanism into a denial-of-service attack vector, leading to violations of service-level agreements for time to first token. We propose RepetitionCurse, a low-cost black-box strategy to exploit this vulnerability. By identifying a universal flaw in MoE router behavior, RepetitionCurse constructs adversarial prompts using simple repetitive token patterns in a model-agnostic manner. On widely deployed MoE models like Mixtral-8x7B, our method increases end-to-end inference latency by 3.063x, degrading service availability significantly.

摘要: 专家混合架构因其卓越的参数效率而成为扩展大型语言模型的标准。为了适应实践中越来越多的专家，现代推理系统通常采用专家并行性来跨设备分布专家。然而，推理过程中缺乏显式的负载平衡约束，导致对抗性输入触发严重的路由集中。我们证明，分发外提示可以操纵路由策略，以便所有令牌一致地路由到同一组顶级k$专家，这在某些设备上造成了计算瓶颈，同时迫使其他设备闲置。这将效率机制转化为拒绝服务攻击载体，导致违反第一个令牌时间的服务级别协议。我们提出了RepetitionCurse，这是一种利用此漏洞的低成本黑匣子策略。通过识别MoE路由器行为中的普遍缺陷，RepetitionCurse以模型不可知的方式使用简单的重复令牌模式构建对抗性提示。在Mixtral-8x 7 B等广泛部署的MoE模型上，我们的方法将端到端推理延迟增加了3.063倍，从而显着降低了服务可用性。



## **11. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

T2 VAttack：对文本到视频扩散模型的对抗性攻击 cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.

摘要: 文本到视频（T2 V）扩散模型的快速发展推动了从自然语言描述生成高质量、时间连贯的视频方面的显着进步。尽管取得了这些成就，但它们对对抗攻击的脆弱性在很大程度上仍然没有被探索。本文介绍了T2 VAttack，这是从语义和时间角度对T2 V扩散模型的对抗性攻击的全面研究。考虑到视频数据固有的动态性质，我们提出了两个不同的攻击目标：评估视频-文本对齐的语义目标和评估时间动态的时间目标。为了实现有效且高效的攻击过程，我们提出了两种对抗攻击方法：（i）T2 VAttack-S，它识别提示中的语义或时间关键词，并通过贪婪搜索用同义词替换它们，和（ii）T2 VAttack-I，它迭代地插入优化的词，对提示的干扰最小。通过结合这些目标和策略，我们对几种最先进的T2 V模型（包括Model Scope、CogVideoX、Open-Sora和HunyuanVideo）的对抗稳健性进行了全面评估。我们的实验表明，即使是微小的即时修改，例如替换或插入单个单词，也可能导致语义保真度和时间动态性的大幅下降，凸显了当前T2 V扩散模型中的关键漏洞。



## **12. Breaking Audio Large Language Models by Attacking Only the Encoder: A Universal Targeted Latent-Space Audio Attack**

仅通过攻击编码器来破解音频大型语言模型：一种通用目标潜在空间音频攻击 cs.SD

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23881v1) [paper-pdf](https://arxiv.org/pdf/2512.23881v1)

**Authors**: Roee Ziv, Raz Lapid, Moshe Sipper

**Abstract**: Audio-language models combine audio encoders with large language models to enable multimodal reasoning, but they also introduce new security vulnerabilities. We propose a universal targeted latent space attack, an encoder-level adversarial attack that manipulates audio latent representations to induce attacker-specified outputs in downstream language generation. Unlike prior waveform-level or input-specific attacks, our approach learns a universal perturbation that generalizes across inputs and speakers and does not require access to the language model. Experiments on Qwen2-Audio-7B-Instruct demonstrate consistently high attack success rates with minimal perceptual distortion, revealing a critical and previously underexplored attack surface at the encoder level of multimodal systems.

摘要: 音频语言模型将音频编码器与大型语言模型相结合，以实现多模式推理，但它们也引入了新的安全漏洞。我们提出了一种通用的有针对性的潜在空间攻击，这是一种编码器级对抗攻击，它操纵音频潜在表示以在下游语言生成中引发攻击者指定的输出。与之前的波级或特定于输入的攻击不同，我们的方法学习了一种通用扰动，该扰动在输入和说话者之间进行概括，并且不需要访问语言模型。Qwen 2-Audio-7 B-Direct上的实验证明，攻击成功率始终很高，感知失真最小，揭示了多模式系统编码器级的关键且之前未充分探索的攻击表面。



## **13. Adversarial Lens: Exploiting Attention Layers to Generate Adversarial Examples for Evaluation**

对抗性镜头：利用注意力层生成对抗性示例进行评估 cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23837v1) [paper-pdf](https://arxiv.org/pdf/2512.23837v1)

**Authors**: Kaustubh Dhole

**Abstract**: Recent advances in mechanistic interpretability suggest that intermediate attention layers encode token-level hypotheses that are iteratively refined toward the final output. In this work, we exploit this property to generate adversarial examples directly from attention-layer token distributions. Unlike prompt-based or gradient-based attacks, our approach leverages model-internal token predictions, producing perturbations that are both plausible and internally consistent with the model's own generation process. We evaluate whether tokens extracted from intermediate layers can serve as effective adversarial perturbations for downstream evaluation tasks. We conduct experiments on argument quality assessment using the ArgQuality dataset, with LLaMA-3.1-Instruct-8B serving as both the generator and evaluator. Our results show that attention-based adversarial examples lead to measurable drops in evaluation performance while remaining semantically similar to the original inputs. However, we also observe that substitutions drawn from certain layers and token positions can introduce grammatical degradation, limiting their practical effectiveness. Overall, our findings highlight both the promise and current limitations of using intermediate-layer representations as a principled source of adversarial examples for stress-testing LLM-based evaluation pipelines.

摘要: 机械解释性的最新进展表明，中间注意力层编码了代币级假设，这些假设经过迭代改进以获得最终输出。在这项工作中，我们利用这一属性直接从注意力层令牌分布生成对抗示例。与基于预算或基于梯度的攻击不同，我们的方法利用模型内部令牌预测，产生既合理又与模型自己的生成过程内部一致的扰动。我们评估从中间层提取的令牌是否可以作为下游评估任务的有效对抗扰动。我们使用ArgQuality数据集进行参数质量评估实验，LLaMA-3.1-Instruct-8B同时作为生成器和评估器。我们的研究结果表明，基于注意力的对抗性示例会导致评估性能的可测量下降，同时保持与原始输入的语义相似。然而，我们也观察到，从某些层和标记位置提取的替代可以引入语法退化，限制其实际效果。总体而言，我们的研究结果突出了使用中间层表示作为基于LLM的评估管道压力测试对抗性示例的原则来源的承诺和当前限制。



## **14. Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems**

安全IIoT防御系统的零信任抽象联邦学习 cs.LG

9 Pages and 6 figures, Submitted in conference 2nd IEEE Conference on Secure and Trustworthy Cyber Infrastructure for IoT and Microelectronics, Houston TX, USA

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23809v1) [paper-pdf](https://arxiv.org/pdf/2512.23809v1)

**Authors**: Samaresh Kumar Singh, Joyjit Roy, Martin So

**Abstract**: Recent attacks on critical infrastructure, including the 2021 Oldsmar water treatment breach and 2023 Danish energy sector compromises, highlight urgent security gaps in Industrial IoT (IIoT) deployments. While Federated Learning (FL) enables privacy-preserving collaborative intrusion detection, existing frameworks remain vulnerable to Byzantine poisoning attacks and lack robust agent authentication. We propose Zero-Trust Agentic Federated Learning (ZTA-FL), a defense in depth framework combining: (1) TPM-based cryptographic attestation achieving less than 0.0000001 false acceptance rate, (2) a novel SHAP-weighted aggregation algorithm providing explainable Byzantine detection under non-IID conditions with theoretical guarantees, and (3) privacy-preserving on-device adversarial training. Comprehensive experiments across three IDS benchmarks (Edge-IIoTset, CIC-IDS2017, UNSW-NB15) demonstrate that ZTA-FL achieves 97.8 percent detection accuracy, 93.2 percent accuracy under 30 percent Byzantine attacks (outperforming FLAME by 3.1 percent, p less than 0.01), and 89.3 percent adversarial robustness while reducing communication overhead by 34 percent. We provide theoretical analysis, failure mode characterization, and release code for reproducibility.

摘要: 最近针对关键基础设施的攻击，包括2021年Oldsmar水处理漏洞和2023年丹麦能源部门的妥协，凸显了工业物联网（IIoT）部署中紧迫的安全漏洞。虽然联邦学习（FL）支持保护隐私的协作入侵检测，但现有框架仍然容易受到拜占庭中毒攻击并且缺乏强大的代理身份验证。我们提出了零信任统计联邦学习（ZTA-FL），这是一种深度防御框架，结合了以下内容：（1）基于BPM的加密证明，实现低于0.0000001的错误接受率，（2）一种新颖的SHAP加权聚合算法，在非IID条件下提供可解释的拜占庭检测，并具有理论保证，以及（3）保护隐私的设备上对抗训练。三个IDS基准测试（Edge-IIoTset、CIC-IDS 2017、UNSW-NB 15）的综合实验表明，ZTA-FL在30%的Byzantine攻击下实现了97.8%的检测准确率、93.2%的准确率（优于FLAME 3.1%，p小于0.01）和89.3%的对抗鲁棒性，同时将通信费用减少34%。我们提供理论分析、故障模式特征和重现性的发布代码。



## **15. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

对基于LLM的学术评论的多语言隐藏提示注入攻击 cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.

摘要: 大型语言模型（LLM）越来越多地被考虑用于高影响力的工作流程，包括学术同行评审。然而，LLM很容易受到文档级隐藏提示注入攻击。在这项工作中，我们构建了一个由ICML接受的大约500篇真实学术论文组成的数据集，并评估在这些文档中嵌入隐藏的对抗提示的效果。每份论文都注入了四种不同语言的语义等效指令，并使用LLM进行审查。我们发现，及时注射会导致英语、日语和中文注射的审查分数和接受/拒绝决定发生重大变化，而阿拉伯语注射几乎没有影响。这些结果凸显了基于LLM的审查系统对文档级提示注入的敏感性，并揭示了不同语言之间脆弱性的显着差异。



## **16. RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking**

RobustMass：通过随机掩蔽来对抗性神经排名攻击的鲁棒性 cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23307v1) [paper-pdf](https://arxiv.org/pdf/2512.23307v1)

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang

**Abstract**: Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems.

摘要: 神经排名模型已经取得了显着的进展，现已广泛部署在现实世界的应用中，例如检索增强生成（RAG）。然而，与其他神经架构一样，它们仍然容易受到对抗性操纵：微妙的字符、单词或短语级扰动可能会毒害检索结果并人为地促进目标候选，从而破坏搜索引擎和下游系统的完整性。现有的防御要么依赖于概括性较差的启发式方法，要么依赖于假设过于强大的对抗性知识的认证方法，从而限制了其实际使用。为了应对这些挑战，我们提出了RobustMasking，这是一种新型防御，将预训练语言模型的上下文预测能力与基于随机掩蔽的平滑机制相结合。我们的方法增强了神经排名模型，以对抗字符、单词和短语级别的对抗性扰动。利用排名模型的成对比较能力和概率统计分析，我们提供了RobustMass认证的Top K稳健性的理论证明。广泛的实验进一步表明，RobustMass成功认证了前10名排名中超过20%的候选文档，免受影响多达30%内容的对抗性干扰。这些结果凸显了RobustMass在增强神经排名模型对抗鲁棒性方面的有效性，标志着朝着为现实世界的检索系统提供更强的安全保障迈出了重要一步。



## **17. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

这是一个陷阱！任务重定向代理Web代理说服基准 cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

摘要: 由大型语言模型支持的基于Web的代理越来越多地用于电子邮件管理或专业网络等任务。然而，它们对动态网络内容的依赖使它们容易受到提示注入攻击：隐藏在界面元素中的对抗指令，说服代理从其原始任务转移。我们介绍了任务重定向代理说服基准（TRAP），这是一项评估，旨在研究说服技术如何在现实任务中误导自主网络代理。在六个前沿模型中，代理人平均容易在25%的任务中立即注入（GPT-5为13%，DeepSeek-R1为43%），微小的界面或上下文变化通常会使成功率翻倍，并揭示了基于网络的代理中系统性、心理驱动的漏洞。我们还提供了一个模块化的社会工程注入框架，在高保真网站克隆上进行受控实验，允许进一步的基准扩展。



## **18. DECEPTICON: How Dark Patterns Manipulate Web Agents**

DECEPTICON：暗模式如何操纵Web代理 cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22894v1) [paper-pdf](https://arxiv.org/pdf/2512.22894v1)

**Authors**: Phil Cuvin, Hao Zhu, Diyi Yang

**Abstract**: Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.

摘要: 欺骗性的UI设计在网络上广泛实例化，通常称为黑暗模式，操纵用户执行与其目标不一致的操作。在本文中，我们表明暗模式在引导代理轨迹方面非常有效，对代理稳健性构成了重大风险。为了量化这种风险，我们引入了DECPTICON，这是一种用于隔离测试单个暗图案的环境。DECPTICON包括700个具有黑暗模式的网络导航任务--600个生成任务和100个现实世界任务，旨在衡量描述跟踪成功和黑暗模式有效性。在最先进的代理中，我们发现在超过70%的测试生成和现实世界任务中，黑暗模式成功地将代理轨迹引导到恶意结果，而人类的平均水平为31%。此外，我们发现暗模式有效性与模型大小和测试时推理正相关，使更大、更强大的模型更容易受到影响。针对对抗性攻击的主要对策，包括背景提示和护栏模型，未能持续降低暗模式干预的成功率。我们的研究结果揭示了黑暗模式对网络代理来说是一种潜在且不可减轻的风险，凸显了对操纵性设计的强有力防御的迫切需要。



## **19. Adaptive Trust Consensus for Blockchain IoT: Comparing RL, DRL, and MARL Against Naive, Collusive, Adaptive, Byzantine, and Sleeper Attacks**

区块链物联网的自适应信任共识：比较RL、DRL和MARL与天真、共谋、自适应、拜占庭和休眠攻击 cs.CR

34 pages, 19 figures, 10 tables. Code available at https://github.com/soham-padia/blockchain-iot-trust

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22860v1) [paper-pdf](https://arxiv.org/pdf/2512.22860v1)

**Authors**: Soham Padia, Dhananjay Vaidya, Ramchandra Mangrulkar

**Abstract**: Securing blockchain-enabled IoT networks against sophisticated adversarial attacks remains a critical challenge. This paper presents a trust-based delegated consensus framework integrating Fully Homomorphic Encryption (FHE) with Attribute-Based Access Control (ABAC) for privacy-preserving policy evaluation, combined with learning-based defense mechanisms. We systematically compare three reinforcement learning approaches -- tabular Q-learning (RL), Deep RL with Dueling Double DQN (DRL), and Multi-Agent RL (MARL) -- against five distinct attack families: Naive Malicious Attack (NMA), Collusive Rumor Attack (CRA), Adaptive Adversarial Attack (AAA), Byzantine Fault Injection (BFI), and Time-Delayed Poisoning (TDP). Experimental results on a 16-node simulated IoT network reveal significant performance variations: MARL achieves superior detection under collusive attacks (F1=0.85 vs. DRL's 0.68 and RL's 0.50), while DRL and MARL both attain perfect detection (F1=1.00) against adaptive attacks where RL fails (F1=0.50). All agents successfully defend against Byzantine attacks (F1=1.00). Most critically, the Time-Delayed Poisoning attack proves catastrophic for all agents, with F1 scores dropping to 0.11-0.16 after sleeper activation, demonstrating the severe threat posed by trust-building adversaries. Our findings indicate that coordinated multi-agent learning provides measurable advantages for defending against sophisticated trust manipulation attacks in blockchain IoT environments.

摘要: 保护支持区块链的物联网网络免受复杂的对抗性攻击仍然是一个关键挑战。提出了一种基于信任的委托共识框架，该框架将全同态加密（FHE）和基于属性的访问控制（ABAC）相结合，结合基于学习的防御机制，用于隐私保护策略评估。我们系统地比较了三种强化学习方法-表格Q学习（RL），Deep RL with Dueling Double DQN（DRL）和Multi-Agent RL（MARL）-针对五种不同的攻击家族：天真恶意攻击（NMA），共谋谣言攻击（CRA），自适应对抗攻击（AAA），拜占庭故障注入（BFI）和延时中毒（TDP）。在16节点模拟物联网网络上的实验结果显示了显著的性能差异：MARL在共谋攻击下实现了卓越的检测（F1=0.85 vs. DRL的0.68和RL的0.50），而DRL和MARL在RL失败（F1=0.50）的自适应攻击下都实现了完美的检测（F1 =1.00）。所有特工都成功抵御了拜占庭攻击（F1=1.00）。最重要的是，事实证明，延时中毒攻击对所有特工来说都是灾难性的，休眠激活后F1评分下降至0.11-0.16，这表明建立信任的对手构成了严重威胁。我们的研究结果表明，协调的多代理学习为防御区块链物联网环境中复杂的信任操纵攻击提供了可衡量的优势。



## **20. Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach**

具有无人机可达性分析的可达-避免差异博弈：分解方法 eess.SY

Paper version accepted to the Journal of Guidance, Control, and Dynamics (JGCD)

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22793v1) [paper-pdf](https://arxiv.org/pdf/2512.22793v1)

**Authors**: Minh Bui, Simon Monckton, Mo Chen

**Abstract**: Reach-avoid (RA) games have significant applications in security and defense, particularly for unmanned aerial vehicles (UAVs). These problems are inherently challenging due to the need to consider obstacles, consider the adversarial nature of opponents, ensure optimality, and account for nonlinear dynamics. Hamilton-Jacobi (HJ) reachability analysis has emerged as a powerful tool for tackling these challenges; however, while it has been applied to games involving two spatial dimensions, directly extending this approach to three spatial dimensions is impossible due to high dimensionality. On the other hand, alternative approaches for solving RA games lack the generality to consider games with three spatial dimensions involving agents with non-trivial system dynamics. In this work, we propose a novel framework for dimensionality reduction by decomposing the problem into a horizontal RA sub-game and a vertical RA sub-game. We then solve each sub-game using HJ reachability analysis and consider second-order dynamics that account for the defender's acceleration. To reconstruct the solution to the original RA game from the sub-games, we introduce a HJ-based tracking control algorithm in each sub-game that not only guarantees capture of the attacker but also tracking of the attacker thereafter. We prove the conditions under which the capture guarantees are maintained. The effectiveness of our approach is demonstrated via numerical simulations, showing that the decomposition maintains optimality and guarantees in the original problem. Our methods are also validated in a Gazebo physics simulator, achieving successful capture of quadrotors in three spatial dimensions space for the first time to the best of our knowledge.

摘要: 避免触及（RA）游戏在安全和国防方面有着重要的应用，特别是对于无人机（UFO）。这些问题本质上具有挑战性，因为需要考虑障碍、考虑对手的对抗性、确保最优性并考虑非线性动态。汉密尔顿-雅各比（TJ）可达性分析已成为应对这些挑战的强大工具;然而，虽然它已应用于涉及两个空间维度的游戏，但由于维度较高，将这种方法直接扩展到三个空间维度是不可能的。另一方面，解决RA游戏的替代方法缺乏考虑涉及具有非平凡系统动态的主体的三个空间维度游戏的通用性。在这项工作中，我们提出了一个新颖的降维框架，通过将问题分解为水平RA子博弈和垂直RA子博弈。然后我们使用HJ可达性分析来解决每个子博弈，并考虑考虑防守者加速度的二阶动力学。为了从子游戏中重建原始RA游戏的解决方案，我们在每个子游戏中引入了一种基于TJ的跟踪控制算法，不仅保证捕获攻击者，而且还保证随后跟踪攻击者。我们证明维持捕获保证的条件。通过数值模拟证明了我们方法的有效性，表明分解保持了最优性并保证了原始问题。我们的方法还在Gazebo物理模拟器中得到了验证，据我们所知，首次实现了在三维空间中成功捕获四螺旋桨。



## **21. Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks**

对尖峰神经网络的对抗鲁棒性进行可靠评估 cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22522v1) [paper-pdf](https://arxiv.org/pdf/2512.22522v1)

**Authors**: Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng

**Abstract**: Spiking Neural Networks (SNNs) utilize spike-based activations to mimic the brain's energy-efficient information processing. However, the binary and discontinuous nature of spike activations causes vanishing gradients, making adversarial robustness evaluation via gradient descent unreliable. While improved surrogate gradient methods have been proposed, their effectiveness under strong adversarial attacks remains unclear. We propose a more reliable framework for evaluating SNN adversarial robustness. We theoretically analyze the degree of gradient vanishing in surrogate gradients and introduce the Adaptive Sharpness Surrogate Gradient (ASSG), which adaptively evolves the shape of the surrogate function according to the input distribution during attack iterations, thereby enhancing gradient accuracy while mitigating gradient vanishing. In addition, we design an adversarial attack with adaptive step size under the $L_\infty$ constraint-Stable Adaptive Projected Gradient Descent (SA-PGD), achieving faster and more stable convergence under imprecise gradients. Extensive experiments show that our approach substantially increases attack success rates across diverse adversarial training schemes, SNN architectures and neuron models, providing a more generalized and reliable evaluation of SNN adversarial robustness. The experimental results further reveal that the robustness of current SNNs has been significantly overestimated and highlighting the need for more dependable adversarial training methods.

摘要: 尖峰神经网络（SNN）利用基于尖峰的激活来模拟大脑的节能信息处理。然而，尖峰激活的二元和不连续性质会导致梯度消失，从而使得通过梯度下降进行的对抗鲁棒性评估不可靠。虽然已经提出了改进的替代梯度方法，但它们在强对抗攻击下的有效性仍不清楚。我们提出了一个更可靠的框架来评估SNN对抗稳健性。我们从理论上分析了代理梯度中梯度消失的程度，并引入了自适应Shareptium Surrogate Gradient（ASSG），它根据攻击迭代期间的输入分布自适应地进化代理函数的形状，从而在减轻梯度消失的同时提高了梯度准确性。此外，我们在$L_\infty$ constraint-Stable Adaptive Projected Gradient Down（SA-PVD）下设计了一种具有自适应步进大小的对抗攻击，在不精确的梯度下实现更快、更稳定的收敛。大量实验表明，我们的方法大大提高了各种对抗训练方案、SNN架构和神经元模型的攻击成功率，为SNN对抗鲁棒性提供了更普遍和可靠的评估。实验结果进一步表明，当前SNN的稳健性被显着高估，并凸显了对更可靠的对抗训练方法的需求。



## **22. NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization**

NOWA：用于隐形捕获指纹识别和篡改定位的零空间光学水印 cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22501v1) [paper-pdf](https://arxiv.org/pdf/2512.22501v1)

**Authors**: Edwin Vargas

**Abstract**: Ensuring the authenticity and ownership of digital images is increasingly challenging as modern editing tools enable highly realistic forgeries. Existing image protection systems mainly rely on digital watermarking, which is susceptible to sophisticated digital attacks. To address this limitation, we propose a hybrid optical-digital framework that incorporates physical authentication cues during image formation and preserves them through a learned reconstruction process. At the optical level, a phase mask in the camera aperture produces a Null-space Optical Watermark (NOWA) that lies in the Null Space of the imaging operator and therefore remains invisible in the captured image. Then, a Null-Space Network (NSN) performs measurement-consistent reconstruction that delivers high-quality protected images while preserving the NOWA signature. The proposed design enables tamper localization by projecting the image onto the camera's null space and detecting pixel-level inconsistencies. Our design preserves perceptual quality, resists common degradations such as compression, and establishes a structural security asymmetry: without access to the optical or NSN parameters, adversaries cannot forge the NOWA signature. Experiments with simulations and a prototype camera demonstrate competitive performance in terms of image quality preservation, and tamper localization accuracy compared to state-of-the-art digital watermarking and learning-based authentication methods.

摘要: 确保数字图像的真实性和所有权越来越具有挑战性，因为现代编辑工具可以实现高度逼真的图像。现有的图像保护系统主要依赖于数字水印技术，但数字水印技术容易受到复杂的数字攻击。为了解决这一限制，我们提出了一个混合的光学数字框架，在图像形成过程中结合了物理认证线索，并通过学习重建过程保留它们。在光学层面上，相机孔径中的相位掩模产生位于成像算子的光空间中的零空间光学水印（NOWA），因此在捕获的图像中保持不可见。然后，空空间网络（NSN）执行测量一致的重建，提供高质量的受保护图像，同时保留NOWA签名。提出的设计通过将图像投影到相机的零空间并检测像素级不一致来实现篡改定位。我们的设计保留了感知质量，抵抗压缩等常见降级，并建立了结构性安全不对称：如果不访问光学或NSN参数，对手就无法伪造NOWA签名。模拟和原型相机的实验表明，与最先进的数字水印和基于学习的认证方法相比，在图像质量保存和篡改定位准确性方面具有竞争力。



## **23. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

PHANTOM：针对联邦学习协调电动汽车充电管理系统的物理感知对抗攻击 cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.

摘要: 配电网内电动汽车充电站（EVCS）的快速部署需要智能和自适应控制，以维持电网的弹性和可靠性。在这项工作中，我们提出了PHANTOM，这是一个物理感知的对抗网络，通过多智能体强化学习模型进行训练和优化。PHANTOM集成了一个由联邦学习（FL）支持的物理信息神经网络（PINN），该网络充当ECVS集成系统的数字孪生体，确保操作动态和约束的物理一致建模。在这个数字双胞胎的基础上，我们构建了一个多智能体RL环境，该环境利用深度Q网络（DQN）和软行动者评论家（SAC）方法来推导能够绕过传统检测机制的对抗性虚假数据注入（Direct）策略。为了检查更广泛的电网级后果，开发了输电和配电（T和D）双重模拟平台，使我们能够捕捉配电级EVCS干扰与批量输电系统运行之间的级联相互作用。结果展示了学习到的攻击策略如何破坏负载平衡并引发跨越T和D边界传播的电压不稳定性。这些发现凸显了对物理感知网络安全的迫切需要，以确保大规模车网集成的弹性。



## **24. Scaling Adversarial Training via Data Selection**

通过数据选择扩展对抗训练 cs.LG

6 pages. Conference workshop paper

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22069v1) [paper-pdf](https://arxiv.org/pdf/2512.22069v1)

**Authors**: Youran Ye, Dejin Wang, Ajinkya Bhandare

**Abstract**: Projected Gradient Descent (PGD) is a strong and widely used first-order adversarial attack, yet its computational cost scales poorly, as all training samples undergo identical iterative inner-loop optimization despite contributing unequally to robustness. Motivated by this inefficiency, we propose \emph{Selective Adversarial Training}, which perturbs only a subset of critical samples in each minibatch. Specifically, we introduce two principled selection criteria: (1) margin-based sampling, which prioritizes samples near the decision boundary, and (2) gradient-matching sampling, which selects samples whose gradients align with the dominant batch optimization direction. Adversarial examples are generated only for the selected subset, while the remaining samples are trained cleanly using a mixed objective. Experiments on MNIST and CIFAR-10 show that the proposed methods achieve robustness comparable to, or even exceeding, full PGD adversarial training, while reducing adversarial computation by up to $50\%$, demonstrating that informed sample selection is sufficient for scalable adversarial robustness.

摘要: 投影梯度下降（PGD）是一种强大且广泛使用的一阶对抗攻击，但其计算成本很低，因为所有训练样本都经过相同的迭代内环优化，尽管对鲁棒性的贡献不一样。出于这种低效率的动机，我们提出了\n {选择性对抗训练}，它只扰动每个小批量中的关键样本的子集。具体来说，我们引入了两个原则性的选择标准：（1）基于边缘的采样，它优先考虑决策边界附近的样本，以及（2）梯度匹配采样，它选择梯度与主要批次优化方向一致的样本。对抗性示例仅针对所选子集生成，而其余样本则使用混合目标进行干净训练。MNIST和CIFAR-10上的实验表明，所提出的方法实现了与完整PVD对抗训练相当甚至超过的鲁棒性，同时将对抗计算减少了高达50%$，这表明明智的样本选择足以实现可扩展的对抗鲁棒性。



## **25. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

很少有令牌重要：对视觉语言模型的熵引导攻击 cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.

摘要: 视觉语言模型（VLM）取得了出色的性能，但仍然容易受到对抗攻击。模型不确定性的衡量指标--与VLM的可靠性密切相关。先前的基于信息量的攻击最大化了所有解码步骤的不确定性，隐含地假设每个令牌对生成不稳定性的贡献相同。相反，我们表明一小部分（约20%）高熵代币，即自回归生成中的关键决策点不成比例地控制着产出轨迹。通过将对抗性扰动集中在这些位置上，我们实现了与全球方法相当的语义降级，同时使用更少的预算。更重要的是，在多个代表性的VLM中，此类选择性攻击将35-49%的良性输出转化为有害输出，暴露了更严重的安全风险。值得注意的是，这些脆弱的高熵分叉会在架构多样的VLM中重复出现，从而实现了可行的可转移性（对不可见目标的有害率为17-26%）。受这些发现的启发，我们提出了熵库引导的对抗攻击（EGA），它实现了竞争性攻击成功率（93-95%）以及高有害转换，从而揭示了当前VLM安全机制的新弱点。



## **26. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

LLM驱动的Android恶意软件检测器的冲突级对抗攻击 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.

摘要: Android恶意软件规模和复杂性的快速增长推动了机器学习（ML）技术的广泛采用，以进行可扩展和准确的恶意软件检测。尽管它们有效，但这些模型仍然容易受到对抗攻击，这些攻击引入精心设计的功能级扰动，以逃避检测，同时保留恶意功能。在本文中，我们介绍了LAMRAD，这是一种新型的对抗性攻击框架，它利用大型语言模型（LLM）的生成和推理能力来绕过基于ML的Android恶意软件分类器。LAMLAT采用双代理架构，由LLM操纵器和LLM分析器组成，LLM操纵器生成真实且功能保留的特征扰动，LLM分析器引导扰动过程成功规避。为了提高效率和上下文感知，LAMRAD将检索增强生成（RAG）集成到LLM管道中。LAMRAD专注于Drebin风格的特征表示，能够针对广泛部署的Android恶意软件检测系统进行隐蔽且高可信度的攻击。我们针对三种代表性的基于ML的Android恶意软件检测器评估LAMRAD，并将其性能与两种最先进的对抗攻击方法进行比较。实验结果表明，LAMRAD的攻击成功率（ASB）高达97%，每个对抗样本平均只需尝试三次，凸显了其在实际对抗环境中的有效性、效率和适应性。此外，我们提出了一种基于对抗训练的防御策略，平均将ASR降低了30%以上，显著增强了模型对LAMLAD攻击的鲁棒性。



## **27. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

CoTDeceptor：针对CoT增强LLM代码代理的对抗代码混淆 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.

摘要: 基于LLM的代码代理（例如，ChatGPT Codex）越来越多地被部署为代码审查和安全审计任务的检测器。尽管CoT增强型LLM漏洞检测器被认为可以针对混淆的恶意代码提供更好的鲁棒性，但我们发现它们的推理链和语义抽象过程表现出可利用的系统弱点。这使得攻击者能够秘密嵌入恶意逻辑、绕过代码审查并在整个现实世界的软件供应链中传播后门组件。为了研究这个问题，我们提出了CoTDeceptor，第一个针对CoT增强型LLM检测器的对抗代码混淆框架。CoTDeceptor自主构建不断发展的、难以逆转的多阶段混淆策略链，有效扰乱CoT驱动的检测逻辑。我们获得了安全企业提供的恶意代码，实验结果表明CoTDeceptor针对最先进的LLM和漏洞检测代理实现了稳定且可转移的规避性能。CoTDeceptor绕过了15个漏洞类别中的14个，而之前的方法只绕过了2个。我们的研究结果强调了现实世界软件供应链中的潜在风险，并强调了对更强大和可解释的LLM支持的安全分析系统的需求。



## **28. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

提高搜索高效硬标签攻击的射线搜索优化的收敛率 cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.

摘要: 在硬标签黑匣子对抗攻击中，只能访问前1名的预测标签，令人望而却步的查询复杂性对实际部署构成了主要障碍。在本文中，我们重点优化一类代表性攻击，这些攻击搜索最佳射线方向，产生将良性图像移动到对抗区域所需的最小$\ell_2 $-norm扰动。受Nesterov加速梯度（NAG）的启发，我们提出了一种基于动量的算法ARS-OPT，该算法主动估计相对于从累积动量推断的未来射线方向的梯度。我们对其收敛行为进行了理论分析，表明ARS-OPT能够实现更准确的方向更新，并实现更快、更稳定的优化。为了进一步加速收敛，我们将代理模型先验纳入ARS-OPT的梯度估计中，从而产生性能增强的PARS-OPT。我们方法的优越性得到了标准假设下的理论保证的支持。ImageNet和CIFAR-10上的大量实验表明，我们的方法在查询效率方面超过了13种最先进的方法。



## **29. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

分时段平衡记录：资源受限系统的有界存储短暂令牌 cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.

摘要: 具有生存时间（TLR）语义的可替代代币需要跟踪每个存入单位的单独到期时间。天真的实施会为每次存款创建新的余额记录，从而导致存储无限增长并容易受到拒绝服务攻击。我们提供分时段的余额记录，这是一种数据结构，将每个帐户的存储限制为O（k）个记录，同时保证令牌不会在其配置的TLR之前到期。我们的方法将时间离散化到k个桶中，将存款合并在同一桶中以限制唯一的到期时间戳。我们证明了三个关键属性：（1）无论存款频率如何，存储都以k+1条记录为界限，（2）实际到期时间始终至少为配置的TLR，以及（3）对手不能将受害者的操作成本增加到O（k）以上[摊销]最坏情况。我们在Solidity中提供了一个参考实施，其中测量的天然气成本证明了实际效率。



## **30. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

模仿游戏：使用大型语言模型作为聊天机器人来打击基于聊天的网络犯罪 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.

摘要: 基于聊天的网络犯罪已成为一种普遍存在的威胁，攻击者利用实时消息平台来实施依赖于信任建立、欺骗和心理操纵的诈骗。传统防御机制基于静态规则或浅层内容过滤器，难以识别这些对话威胁，尤其是当攻击者使用多媒体混淆和上下文感知对话时。   在这部作品中，我们提出了一个受经典模仿游戏启发的挑衅性问题：机器能否令人信服地冒充人类受害者，利用欺骗手段对付网络犯罪分子？我们介绍了LURE（基于LLM的用户响应参与），这是第一个将大型语言模型（LLM）部署为嵌入在对抗性聊天环境中的主动代理而不是被动分类器的系统。   LURE结合了自动发现、对抗交互和基于OCR的图像嵌入式支付数据分析。应用于Telegram上的非法视频聊天诈骗设置，我们的系统涉及98个群组的53名参与者。在超过56%的互动中，LLM保持多轮对话，而不会被机器人注意到，有效地“赢得”了模仿游戏。我们的调查结果揭示了诈骗操作中的关键行为模式，例如支付流、向上销售策略和平台迁移策略。



## **31. Robustness Certificates for Neural Networks against Adversarial Attacks**

神经网络抗对抗性攻击的鲁棒性证明 cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.

摘要: 机器学习在安全关键领域的使用越来越多，放大了对抗威胁的风险，特别是破坏训练数据以降低性能或引发不安全行为的数据中毒攻击。大多数现有的防御缺乏正式保证或依赖于有关模型类别、攻击类型、中毒程度或逐点认证的限制性假设，从而限制了其实际可靠性。本文介绍了一个有原则的正式鲁棒性认证框架，该框架将基于梯度的训练建模为离散时间动态系统（dt-DS），并将中毒鲁棒性制定为正式安全验证问题。通过改编来自控制理论的屏障证书（BC）概念，我们引入了充分条件来证明稳健半径，以确保终端模型在最坏情况下${\ell}_p$-norm基于中毒的情况下保持安全。为了实现这一点，我们将BC参数化为在有限组中毒轨迹上训练的神经网络。我们进一步通过求解场景凸规划（SCP）来推导出可能大致正确（PAC）界限，这会产生扩展到训练集之外的认证稳健性半径的置信下限。重要的是，我们的框架还扩展到针对测试时攻击的认证，使其成为第一个在训练和测试时攻击环境中提供正式保证的统一框架。MNIST、SVHN和CIFAR-10上的实验表明，我们的方法可以证明非平凡的扰动预算，同时是模型不可知的，并且不需要攻击或污染水平的先验知识。



## **32. Defending against adversarial attacks using mixture of experts**

使用混合专家抵御对抗攻击 cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.

摘要: 机器学习是一种强大的工具，无需显式编程即可实现大量任务的完全自动化。尽管机器学习最近在不同领域取得了进展，但这些模型在面临对抗威胁时仍表现出脆弱性。对抗性威胁旨在阻碍机器学习模型实现其目标。它们可以产生对抗性扰动，人类肉眼无法察觉，但有能力在推理过程中导致错误分类。此外，他们可以毒害训练数据以损害模型的性能，或者他们可以查询模型以窃取其敏感信息。在本文中，我们提出了一种防御系统，该系统在混合专家架构中设计了一个对抗训练模块，以增强其对对抗威胁的鲁棒性。在我们提出的防御系统中，我们使用九名经过预先培训的专家，以ResNet-18为骨干。在端到端训练过程中，专家模型和门控机制的参数联合更新，从而进一步优化专家。我们提出的防御系统优于最先进的防御系统和普通分类器，后者使用比我们模型的主干更复杂的架构。



## **33. Safety Alignment of LMs via Non-cooperative Games**

通过非合作博弈实现LM的安全调整 cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.

摘要: 确保语言模型（LM）的安全性同时保持其有用性仍然是人工智能协调的一个关键挑战。当前的方法依赖于顺序对抗训练：生成对抗提示并微调LM以抵御它们。我们引入了一种不同的范式：将安全对齐框架为攻击者LM和防御者LM之间通过在线强化学习联合训练的非零和游戏。每个LM都不断适应对方不断发展的策略，推动迭代改进。我们的方法使用基于偏好的奖励信号，而不是逐点比较，提供更强大的监督，并可能减少奖励黑客。我们的RL配方AdvGame改变了安全性和实用性的帕累托边界，产生了一个防御者LM，同时对对抗性攻击更有帮助，更有弹性。此外，由此产生的攻击者LM收敛到一个强大的，通用的红队代理，可以直接部署到探测任意目标模型。



## **34. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

机器人群中无通信协作避碰的基于应急模型的控制（MC） math.OC

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.20391v3) [paper-pdf](https://arxiv.org/pdf/2512.20391v3)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots, or `agents,' in swarm operations remains an open challenge. Assuming a decentralized architecture, each agent is responsible for making its own decisions and choosing its control actions. Most existing approaches rely on a (wireless) communication network between (some of) the agents. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, and transmission faults. Moreover, it is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC), a decentralized cooperative approach that does not rely on communication. Instead, the control algorithm is based on consensual rules that are designed for all agents offline, similar to traffic rules. For CMC, this includes the definition of a contingency trajectory for each robot, and perpendicular bisecting planes as collision avoidance constraints. The setup permits a full guarantee of recursive feasibility and collision avoidance between all swarm members in closed-loop operation. CMC naturally satisfies the plug & play paradigm, i.e., new robots may enter the swarm dynamically. The effectiveness of the CMC regime is demonstrated in two numerical examples, showing that the collision avoidance guarantee is intact and the robot swarm operates smoothly in a constrained environment.

摘要: 群体行动中机器人或“代理人”之间的合作避免碰撞仍然是一个悬而未决的挑战。假设采用去中心化架构，每个代理负责做出自己的决策并选择其控制动作。大多数现有的方法依赖于（一些）代理之间的（无线）通信网络。然而，事实上，沟通是脆弱的。它可能会受到延迟、进一步延迟和数据包丢失以及传输故障的影响。此外，它还会受到对抗攻击，例如干扰或欺骗。本文提出了基于权宜模型的控制（MCC），这是一种不依赖于通信的去中心化合作方法。相反，控制算法基于为所有离线代理设计的共识规则，类似于交通规则。对于MCC，这包括定义每个机器人的应急轨迹，以及垂直平分平面作为碰撞避免约束。该设置可以充分保证闭环操作中所有群成员之间的循环可行性和冲突避免。SMC自然满足即插即用范式，即新的机器人可能会动态地进入群体。通过两个算例验证了CMS机制的有效性，表明避碰保证完好无损，机器人群在受约束的环境中平稳运行。



## **35. GShield: Mitigating Poisoning Attacks in Federated Learning**

GShield：减轻联邦学习中的中毒攻击 cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.19286v2) [paper-pdf](https://arxiv.org/pdf/2512.19286v2)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.

摘要: 联合学习（FL）最近成为协作训练机器学习模型的革命性方法。特别是，它能够实现去中心化模型训练，同时保护数据隐私，但其分布式性质使其极易受到称为数据中毒的严重攻击。在此类情况下，恶意客户端将操纵数据注入到训练过程中，从而降低全局模型性能或导致有针对性的错误分类。在本文中，我们提出了一种名为GShield的新型防御机制，旨在检测和减轻恶意和低质量更新，特别是在非独立和同分布（非IID）数据场景下。GShield通过在初始一轮期间通过集群和高斯建模学习良性梯度的分布来运作，使其能够建立可信客户行为的可靠基线。通过这种良性配置文件，GShield选择性地仅聚合那些与预期梯度模式一致的更新，从而有效地隔离敌对客户并保持全球模型的完整性。一项广泛的实验活动表明，与最先进的方法相比，我们提出的防御显着提高了模型的稳健性，同时在表格和图像数据集中保持了高准确性的性能。此外，GShield在检测到恶意和低质量客户端后将目标类的准确性提高了43%至65%。



## **36. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **37. Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN**

亲眼目睹并不可信：通过条件GAN的上下文感知对抗补丁合成 cs.CV

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2509.22836v2) [paper-pdf](https://arxiv.org/pdf/2509.22836v2)

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.   Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness.

摘要: 对抗性补丁攻击对深度神经网络构成严重威胁，但大多数现有方法依赖于不切实际的白盒假设、无针对性的目标，或产生视觉上明显的补丁，从而限制了现实世界的适用性。在这项工作中，我们引入了一种新颖的框架，用于完全可控的对抗补丁生成，其中攻击者可以自由选择输入图像x和目标类别y目标，从而决定确切的误分类结果。我们的方法将生成式U-Net设计与Grad-CAM引导的补丁放置相结合，实现语义感知的本地化，从而最大限度地提高攻击效果，同时保持视觉真实感。跨卷积网络（DenseNet-121、ResNet-50）和视觉转换器（ViT-B/16、Swin-B/16等）的广泛实验表明，我们的方法在所有设置中都实现了最先进的性能，攻击成功率（ASB）和目标级成功率（TCS）始终超过99%。   重要的是，我们表明我们的方法不仅优于之前的白盒攻击和无针对性基线，而且还优于现有的产生可检测伪影的非现实方法。通过同时确保真实性、有针对性的控制和黑匣子适用性（基于补丁的攻击中最具挑战性的三个方面），我们的框架为对抗稳健性研究建立了新的基准，弥合了理论攻击强度和实际隐蔽性之间的差距。



## **38. BadBlocks: Lightweight and Stealthy Backdoor Threat in Text-to-Image Diffusion Models**

BadBlocks：文本到图像扩散模型中的轻量级且隐蔽的后门威胁 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2508.03221v4) [paper-pdf](https://arxiv.org/pdf/2508.03221v4)

**Authors**: Yu Pan, Jiahao Chen, Wenjie Wang, Bingrong Dai, Junjun Yang

**Abstract**: Diffusion models have recently achieved remarkable success in image generation, yet growing evidence shows their vulnerability to backdoor attacks, where adversaries implant covert triggers to manipulate outputs. While existing defenses can detect many such attacks via visual inspection and neural network-based analysis, we identify a more lightweight and stealthy threat, termed BadBlocks. BadBlocks selectively contaminates specific blocks within the UNet architecture while preserving the normal behavior of the remaining components. Compared with prior methods, it requires only about 30% of the computation and 20% of the GPU time, yet achieves high attack success rates with minimal perceptual degradation. Extensive experiments demonstrate that BadBlocks can effectively evade state-of-the-art defenses, particularly attention-based detection frameworks. Ablation studies further reveal that effective backdoor injection does not require fine-tuning the entire network and highlight the critical role of certain layers in backdoor mapping. Overall, BadBlocks substantially lowers the barrier for backdooring large-scale diffusion models, even on consumer-grade GPUs.

摘要: 扩散模型最近在图像生成方面取得了显着的成功，但越来越多的证据表明它们容易受到后门攻击，即对手植入秘密触发器来操纵输出。虽然现有的防御系统可以通过视觉检查和基于神经网络的分析来检测许多此类攻击，但我们识别出了一种更轻量级、更隐蔽的威胁，称为BadBlocks。BadBlocks选择性地污染UNet架构中的特定块，同时保留其余组件的正常行为。与现有方法相比，它只需要约30%的计算量和20%的图形处理时间，但在感知退化最小的情况下实现了高攻击成功率。大量实验表明，BadBlocks可以有效地规避最先进的防御，尤其是基于注意力的检测框架。消融研究进一步表明，有效的后门注入不需要微调整个网络，并强调某些层在后门映射中的关键作用。总体而言，BadBlocks大大降低了大规模扩散模型的后门障碍，即使是在消费级图形处理器上也是如此。



## **39. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

BeDKD：基于动态知识蒸馏和方向映射调制器的后门防御 cs.CR

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2508.01595v2) [paper-pdf](https://arxiv.org/pdf/2508.01595v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.

摘要: 尽管现有的后门防御措施在缓解后门攻击方面取得了成功，但它们仍然面临着巨大的挑战。特别是，它们中的大多数依赖大量干净的数据来削弱后门映射，但通常会与残余触发效应作斗争，从而导致攻击成功率（ASB）持续很高。因此，本文提出了一种基于方向映射模块和对抗性知识蒸馏（BeDKD）的新型后门防御方法，该方法使用少量干净和有毒数据来平衡防御有效性和模型性能之间的权衡。我们首先引入一个方向性映射模块来识别有毒数据，这会破坏干净映射，同时在一小群翻转干净数据上保留后门映射。然后，对抗性知识蒸馏旨在通过信任和惩罚蒸馏之间的循环迭代机制来加强干净映射并抑制后门映射，使用干净和已识别的有毒数据。我们进行了实验来缓解对三个数据集的主流攻击，实验结果表明，BeDKD超越了最先进的防御能力，并在不显着降低CACC的情况下将ASB降低了98%。我们的代码可在https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD上获取。



## **40. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对各种不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难概括不同的攻击类型，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，制定模型防御作为一个对比表示学习（CRL）的问题。我们的方法使用基于三元组的损失结合对抗性硬负面挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **41. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

CAE-Net：使用具有空间和频域特征的卷积和注意力机制的广义Deepfake图像检测 cs.CV

Published in Journal of Visual Communication and Image Representation

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2502.10682v3) [paper-pdf](https://arxiv.org/pdf/2502.10682v3)

**Authors**: Anindya Bhattacharjee, Kaidul Islam, Kafi Anan, Ashir Intesher, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: The spread of deepfakes poses significant security concerns, demanding reliable detection methods. However, diverse generation techniques and class imbalance in datasets create challenges. We propose CAE-Net, a Convolution- and Attention-based weighted Ensemble network combining spatial and frequency-domain features for effective deepfake detection. The architecture integrates EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet features to learn complementary representations. We evaluated CAE-Net on the diverse IEEE Signal Processing Cup 2025 (DF-Wild Cup) dataset, which has a 5:1 fake-to-real class imbalance. To address this, we introduce a multistage disjoint-subset training strategy, sequentially training the model on non-overlapping subsets of the fake class while retaining knowledge across stages. Our approach achieved $94.46\%$ accuracy and a $97.60\%$ AUC, outperforming conventional class-balancing methods. Visualizations confirm the network focuses on meaningful facial regions, and our ensemble design demonstrates robustness against adversarial attacks, positioning CAE-Net as a dependable and generalized deepfake detection framework.

摘要: Deepfakes的传播带来了重大的安全问题，需要可靠的检测方法。然而，多样化的生成技术和数据集的类别不平衡带来了挑战。我们提出CAE-Net，这是一种基于卷积和注意力的加权Ensemble网络，结合了空间和频域特征，用于有效的深度伪造检测。该架构将EfficientNet、数据高效图像Transformer（DeiT）和ConvNeXt与子波功能集成，以学习补充表示。我们在不同的IEEE Signal Process Cup 2025（DF-Wild Cup）数据集上评估了CAE-Net，该数据集的真实与真实类别失衡为5：1。为了解决这个问题，我们引入了一种多阶段不相交子集训练策略，在假类的非重叠子集上顺序训练模型，同时保留跨阶段的知识。我们的方法实现了94.46美元的准确性和97.60美元的UC，优于传统的类别平衡方法。可视化证实该网络专注于有意义的面部区域，我们的整体设计展示了针对对抗攻击的鲁棒性，将CAE-Net定位为可靠且通用的Deepfake检测框架。



## **42. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型（LLM）的检索增强生成（RAG）系统对于问答和内容生成等任务来说已变得至关重要。然而，由于其固有的漏洞，它们对公众舆论和信息传播的影响越来越大，使它们成为安全研究的关键焦点。之前的研究主要针对针对事实或单一查询操纵的攻击。在本文中，我们讨论了一个更实际的场景：对RAG模型的面向主题的对抗性意见操纵攻击，其中LLM需要推理和综合多个观点，使其特别容易受到系统性知识中毒的影响。具体来说，我们提出了Topic-FlipRAG，这是一种两阶段操纵攻击管道，可以战略性地制造对抗性扰动，以影响相关查询的意见。该方法结合了传统的对抗性排名攻击技术，并利用LLM的广泛内部相关知识和推理能力来执行语义级别的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显着影响用户信息感知。当前的缓解方法无法有效防御此类攻击，这凸显了加强RAG系统保护措施的必要性，并为LLM安全研究提供了重要见解。



## **43. Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges**

相关性幻觉：任意内容注入攻击欺骗检索者、重新攻击者和LLM法官 cs.IR

AACL Findings 2025

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2501.18536v2) [paper-pdf](https://arxiv.org/pdf/2501.18536v2)

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: This work considers a black-box threat model in which adversaries attempt to propagate arbitrary non-relevant content in search. We show that retrievers, rerankers, and LLM relevance judges are all highly vulnerable to attacks that enable arbitrary content to be promoted to the top of search results and to be assigned perfect relevance scores. We investigate how attackers may achieve this via content injection, injecting arbitrary sentences into relevant passages or query terms into arbitrary passages. Our study analyzes how factors such as model class and size, the balance between relevant and non-relevant content, injection location, toxicity and severity of injected content, and the role of LLM-generated content influence attack success, yielding novel, concerning, and often counterintuitive results. Our results reveal a weakness in embedding models, LLM-based scoring models, and generative LLMs, raising concerns about the general robustness, safety, and trustworthiness of language models regardless of the type of model or the role in which they are employed. We also emphasize the challenges of robust defenses against these attacks. Classifiers and more carefully prompted LLM judges often fail to recognize passages with content injection, especially when considering diverse text topics and styles. Our findings highlight the need for further research into arbitrary content injection attacks. We release our code for further study.

摘要: 这项工作考虑了黑匣子威胁模型，其中对手试图在搜索中传播任意不相关内容。我们表明，检索者、重新排名者和LLM相关性判断者都极易受到攻击，这些攻击使任意内容能够被提升到搜索结果的顶部并被赋予完美的相关性分数。我们调查攻击者如何通过内容注入来实现这一目标，将任意句子注入相关段落或将查询术语注入任意段落。我们的研究分析了模型类别和大小、相关内容和非相关内容之间的平衡、注入位置、注入内容的毒性和严重性以及LLM生成的内容的作用等因素如何影响攻击成功，从而产生新颖、令人担忧且往往违反直觉的结果。我们的结果揭示了嵌入模型、基于LLM的评分模型和生成式LLM的弱点，引发了人们对语言模型的总体稳健性、安全性和可信性的担忧，无论模型类型或其所扮演的角色如何。我们还强调针对这些攻击的强有力防御所面临的挑战。分类器和更仔细提示的LLM评委通常无法识别具有内容注入的段落，尤其是在考虑不同的文本主题和风格时。我们的研究结果凸显了对任意内容注入攻击进行进一步研究的必要性。我们发布了我们的代码以供进一步研究。



## **44. When Should Selfish Miners Double-Spend?**

什么时候自私的矿工应该加倍花钱？ cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2501.03227v4) [paper-pdf](https://arxiv.org/pdf/2501.03227v4)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.

摘要: 传统的双重支出攻击模型忽略了孤儿区块带来的收入损失。另一方面，自私的采矿文献通常忽视攻击者在每个攻击周期中免费重复支出的机会。在本文中，我们给出了一个严格的随机分析的攻击对手的目标是双花，而自私地挖掘。为此，我们首先结合顽固和自私的采矿攻击，即构建一个策略，让攻击者表现得顽固，直到其私人分支达到一定长度，然后转向自私。我们为每个参数制度提供最佳的确定性。接下来，我们提供了仍然比诚实采矿更有利可图的最大顽固度，并论证了顽固度水平与$k$-确认规则之间的联系。我们表明，在每个攻击周期中，如果顽固程度高于$k$，对手就可以免费获得双重消费的机会。在每个周期中，对于给定的顽固度水平，我们严格制定双重消费的可能性有多大。我们进一步修改顽固政权中的攻击，以隐藏攻击并增加双重消费的概率。



## **45. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

FlippedRAG：黑盒意见操纵对抗性攻击检索增强生成模型 cs.IR

Accepted by 32nd ACM Conference on Computer and Communications Security (ACM CCS 2025)

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2501.02968v6) [paper-pdf](https://arxiv.org/pdf/2501.02968v6)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.

摘要: 检索增强生成（RAG）通过动态检索外部知识、减少幻觉和满足实时信息需求来丰富LLM。虽然现有研究主要针对RAG的性能和效率，但新出现的研究强调了关键的安全问题。然而，当前的对抗方法仍然有限，主要解决白盒场景或启发式黑匣子攻击，而没有在检索阶段充分调查漏洞。此外，之前的作品主要集中在事实问答任务上，其攻击缺乏复杂性，并且可以通过高级LLM轻松纠正。在本文中，我们研究了一个更现实和关键的威胁场景：针对黑箱RAG模型的意见操纵的对抗性攻击，特别是在有争议的话题上。具体来说，我们提出了FlippedRAG，这是一种针对黑盒RAG系统的基于传输的对抗性攻击。我们首先证明了一个黑盒RAG系统的底层检索器可以进行逆向工程，使我们能够训练一个代理检索器。利用代理检索器，我们进一步工艺目标中毒触发器，改变不同的几个文件，以有效地操纵检索和后续生成。广泛的实证结果表明，FlippedRAG的性能大大优于基线方法，将平均攻击成功率提高了16.7%。FlippedRAG平均实现了RAG生成的响应的意见两极50%的方向性转变，最终导致用户认知发生了20%的显着转变。此外，我们评估了几种潜在防御措施的性能，得出的结论是，现有的缓解策略仍然不足以应对此类复杂的操纵攻击。这些结果凸显了开发创新防御解决方案的迫切需要，以确保RAG系统的安全性和可信性。



## **46. Quantifying True Robustness: Synonymity-Weighted Similarity for Trustworthy XAI Evaluation**

量化真正稳健性：可信赖XAI评估的同义加权相似性 cs.LG

10 pages, 2 figures, 6 tables. Changes to title, abstract and minor edits to the content as a result of acceptance to the 59th Hawaii International Conference on System Sciences

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2501.01516v2) [paper-pdf](https://arxiv.org/pdf/2501.01516v2)

**Authors**: Christopher Burger

**Abstract**: Adversarial attacks challenge the reliability of Explainable AI (XAI) by altering explanations while the model's output remains unchanged. The success of these attacks on text-based XAI is often judged using standard information retrieval metrics. We argue these measures are poorly suited in the evaluation of trustworthiness, as they treat all word perturbations equally while ignoring synonymity, which can misrepresent an attack's true impact. To address this, we apply synonymity weighting, a method that amends these measures by incorporating the semantic similarity of perturbed words. This produces more accurate vulnerability assessments and provides an important tool for assessing the robustness of AI systems. Our approach prevents the overestimation of attack success, leading to a more faithful understanding of an XAI system's true resilience against adversarial manipulation.

摘要: 对抗性攻击通过改变解释而模型的输出保持不变，挑战了可解释人工智能（XAI）的可靠性。通常使用标准信息检索指标来判断这些对基于文本的XAI的攻击是否成功。我们认为，这些措施不适合评估可信度，因为它们平等地对待所有单词扰动，而忽略了同义性，这可能会歪曲攻击的真实影响。为了解决这个问题，我们应用了同义加权，这是一种通过合并受干扰单词的语义相似性来修改这些测量的方法。这可以产生更准确的漏洞评估，并为评估人工智能系统的稳健性提供了重要工具。我们的方法可以防止高估攻击成功，从而更忠实地了解XAI系统对抗对抗操纵的真正弹性。



## **47. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

通过对抗稳健性评估改进图神经网络训练、防御和超图划分 cs.LG

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2412.14738v10) [paper-pdf](https://arxiv.org/pdf/2412.14738v10)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.

摘要: 图神经网络（GNN）是一种用于处理图结构数据的高效神经网络架构。与仅依赖数据特征作为输入的传统神经网络不同，GNN利用表示数据点之间关系的图结构和数据的特征矩阵来优化其特征表示。这种独特的功能使GNN能够在各种任务中实现卓越的性能。然而，它也使GNN更容易受到来自图结构和数据特征的噪音和对抗攻击，这可能会显着增加训练难度并降低其性能。同样，超图是一种高度复杂的结构，划分超图是一项具有挑战性的任务。本文利用谱对抗鲁棒性评估来有效解决复杂图算法中的关键挑战。通过使用谱对抗鲁棒性评估来区分鲁棒节点和非鲁棒节点并区别对待它们，我们提出了一种提高GNN训练质量的训练集构建策略。此外，我们还开发了算法来增强GNN的对抗鲁棒性和超图分区的性能。实验结果表明，该系列方法非常有效。



## **48. SoundnessBench: A Soundness Benchmark for Neural Network Verifiers**

SoundnessBench：神经网络验证者的健全基准 cs.LG

TMLR (December 2025)

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2412.03154v3) [paper-pdf](https://arxiv.org/pdf/2412.03154v3)

**Authors**: Xingjian Zhou, Keyi Shen, Andy Xu, Hongji Xu, Cho-Jui Hsieh, Huan Zhang, Zhouxing Shi

**Abstract**: Neural network (NN) verification aims to formally verify properties of NNs, which is crucial for ensuring the behavior of NN-based models in safety-critical applications. In recent years, the community has developed many NN verifiers and benchmarks to evaluate them. However, existing benchmarks typically lack ground-truth for hard instances where no current verifier can verify the property and no counterexample can be found. This makes it difficult to validate the soundness of a verifier, when it claims verification on such challenging instances that no other verifier can handle. In this work, we develop a new benchmark for NN verification, named SoundnessBench, specifically for testing the soundness of NN verifiers. SoundnessBench consists of instances with deliberately inserted counterexamples that are hidden from adversarial attacks commonly used to find counterexamples. Thereby, it can identify false verification claims when hidden counterexamples are known to exist. We design a training method to produce NNs with hidden counterexamples and systematically construct our SoundnessBench with instances across various model architectures, activation functions, and input data. We demonstrate that our training effectively produces hidden counterexamples and our SoundnessBench successfully identifies bugs in state-of-the-art NN verifiers. Our code is available at https://github.com/mvp-harry/SoundnessBench and our dataset is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.

摘要: 神经网络（NN）验证旨在正式验证NN的属性，这对于确保基于NN的模型在安全关键应用中的行为至关重要。近年来，社区开发了许多NN验证器和基准来评估它们。然而，现有的基准测试通常缺乏针对当前验证者可以验证属性且找不到反例的硬实例的基本事实。这使得当验证者声称对没有其他验证者可以处理的具有挑战性的实例进行验证时，很难验证者的合理性。在这项工作中，我们开发了一个新的NN验证基准，名为SoundnessBench，专门用于测试NN验证器的可靠性。SoundnessBench由故意插入反例的实例组成，这些反例隐藏在通常用于查找反例的对抗性攻击中。因此，当已知存在隐藏的反例时，它可以识别错误的验证声明。我们设计了一种训练方法来产生具有隐藏反例的NN，并系统地构建了我们的SoundnessBench，其中包含各种模型架构，激活函数和输入数据的实例。我们证明了我们的训练有效地产生了隐藏的反例，并且我们的SoundnessBench成功地识别了最先进的NN验证器中的错误。我们的代码可在https://github.com/mvp-harry/SoundnessBench上获取，我们的数据集可在https://huggingface.co/datasets/SoundnessBench/SoundnessBench上获取。



## **49. Trust-free Personalized Decentralized Learning**

无需信任的个性化分散学习 cs.LG

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2410.11378v2) [paper-pdf](https://arxiv.org/pdf/2410.11378v2)

**Authors**: Yawen Li, Yan Li, Junping Du, Yingxia Shao, Meiyu Liang, Guanhua Ye

**Abstract**: Personalized collaborative learning in federated settings faces a critical trade-off between customization and participant trust. Existing approaches typically rely on centralized coordinators or trusted peer groups, limiting their applicability in open, trust-averse environments. While recent decentralized methods explore anonymous knowledge sharing, they often lack global scalability and robust mechanisms against malicious peers. To bridge this gap, we propose TPFed, a \textit{Trust-free Personalized Decentralized Federated Learning} framework. TPFed replaces central aggregators with a blockchain-based bulletin board, enabling participants to dynamically select global communication partners based on Locality-Sensitive Hashing (LSH) and peer ranking. Crucially, we introduce an ``all-in-one'' knowledge distillation protocol that simultaneously handles knowledge transfer, model quality evaluation, and similarity verification via a public reference dataset. This design ensures secure, globally personalized collaboration without exposing local models or data. Extensive experiments demonstrate that TPFed significantly outperforms traditional federated baselines in both learning accuracy and system robustness against adversarial attacks.

摘要: 联邦环境中的个性化协作学习面临着定制和参与者信任之间的关键权衡。现有的方法通常依赖于集中式协调员或受信任的同侪团体，限制了它们在开放、厌恶信任的环境中的适用性。虽然最近的去中心化方法探索了匿名知识共享，但它们通常缺乏全球可扩展性和针对恶意对等点的强大机制。为了弥合这一差距，我们提出了TPFed，这是一个\textit{Trust-free Personalized Decentralized Federated Learning}框架。TPFed用基于区块链的公告板取代了中央聚合器，使参与者能够根据本地敏感哈希（LSH）和对等排名动态选择全球通信合作伙伴。至关重要的是，我们引入了一个“一体化”的知识蒸馏协议，同时处理知识转移，模型质量评估，并通过公共参考数据集的相似性验证。这种设计确保了安全、全球个性化的协作，而不会暴露本地模型或数据。大量的实验表明，TPFed在学习准确性和系统对对抗性攻击的鲁棒性方面都显着优于传统的联邦基线。



## **50. Achieving Dalenius' Goal of Data Privacy with Practical Assumptions**

通过实际假设实现Dalenius的数据隐私目标 cs.CR

50 pages

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/1703.07474v6) [paper-pdf](https://arxiv.org/pdf/1703.07474v6)

**Authors**: Genqiang Wu, Xianyao Xia, Yeping He

**Abstract**: Current differential privacy frameworks face significant challenges: vulnerability to correlated data attacks and suboptimal utility-privacy tradeoffs. To address these limitations, we establish a novel information-theoretic foundation for Dalenius' privacy vision using Shannon's perfect secrecy framework. By leveraging the fundamental distinction between cryptographic systems (small secret keys) and privacy mechanisms (massive datasets), we replace differential privacy's restrictive independence assumption with practical partial knowledge constraints ($H(X) \geq b$).   We propose an information privacy framework achieving Dalenius security with quantifiable utility-privacy tradeoffs. Crucially, we prove that foundational mechanisms -- random response, exponential, and Gaussian channels -- satisfy Dalenius' requirements while preserving group privacy and composition properties. Our channel capacity analysis reduces infinite-dimensional evaluations to finite convex optimizations, enabling direct application of information-theoretic tools.   Empirical evaluation demonstrates that individual channel capacity (maximal information leakage of each individual) decreases with increasing entropy constraint $b$, and our framework achieves superior utility-privacy tradeoffs compared to classical differential privacy mechanisms under equivalent privacy guarantees. The framework is extended to computationally bounded adversaries via Yao's theory, unifying cryptographic and statistical privacy paradigms. Collectively, these contributions provide a theoretically grounded path toward practical, composable privacy -- subject to future resolution of the tradeoff characterization -- with enhanced resilience to correlation attacks.

摘要: 当前的差异隐私框架面临着重大挑战：容易受到相关数据攻击以及次优的公用事业与隐私权衡。为了解决这些局限性，我们使用香农的完美保密框架为Dalenius的隐私愿景建立了一个新颖的信息理论基础。通过利用加密系统（小秘密密钥）和隐私机制（大规模数据集）之间的根本区别，我们用实际的部分知识约束（$H（X）\geq b$）取代差异隐私的限制性独立性假设。   我们提出了一个信息隐私框架，通过可量化的公用事业与隐私权衡来实现Dalenius安全。至关重要的是，我们证明了基本机制--随机响应、指数和高斯通道--可以满足Dalenius的要求，同时保留组隐私和组合属性。我们的渠道容量分析将无限维评估简化为有限凸优化，从而能够直接应用信息论工具。   经验评估表明，个人通道容量（每个人的最大信息泄露）随着信息量约束$b$的增加而减少，并且与等效隐私保证下的经典差异隐私机制相比，我们的框架实现了更好的实用性-隐私权衡。该框架通过Yao的理论扩展到计算有界限的对手，统一了加密和统计隐私范式。总的来说，这些贡献为实现实用的、可组合的隐私提供了一条理论上的基础路径--取决于权衡特征的未来解决方案--并增强了对相关性攻击的弹性。



