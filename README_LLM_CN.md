# Latest Large Language Model Attack Papers
**update at 2025-10-24 10:12:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

RAGrank：使用PageRank来应对RTI LLM管道中的中毒 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20768v1) [paper-pdf](http://arxiv.org/pdf/2510.20768v1)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.

摘要: 检索增强生成（RAG）已成为在网络威胁情报（RTI）系统中操作大型语言模型（LLM）使用的主要架构模式。然而，这种设计很容易受到中毒攻击，并且之前提出的防御措施可能会在RTI上下文中失败，因为网络威胁信息对于新兴攻击来说通常是全新的，而且复杂的威胁行为者可以模仿合法的格式、术语和文体惯例。为了解决这个问题，我们建议可以通过在数据库上应用源可信度算法来加速现代RAG防御的鲁棒性，以PageRank为例。在我们的实验中，我们量化地证明，我们的算法使用标准化的MS MARCO数据集，在推广可信内容的同时，将较低的权威分数应用于恶意文档。我们还在RTI文档和提要上展示了我们算法的概念验证性能。



## **2. Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders**

Breaking Bad令牌：使用稀疏自编码器对LLM进行去重编码 cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2505.14536v2) [paper-pdf](http://arxiv.org/pdf/2505.14536v2)

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.

摘要: 大型语言模型（LLM）现在在面向用户的应用程序中无处不在，但它们仍然会产生不受欢迎的有毒输出，包括脏话、粗俗和贬损言论。尽管存在多种解毒方法，但大多数都适用于广泛的、表面的修复，因此很容易被越狱攻击规避。在本文中，我们利用稀疏自动编码器（SAEs）来识别模型剩余流中与毒性相关的方向，并使用相应的解码器载体执行有针对性的激活引导。我们引入了三层转向攻击性，并在GPT-2 Small和Gemma-2-2B上对其进行了评估，揭示了毒性降低和语言流利性之间的权衡。在更强的引导强度下，这些因果干预措施在将毒性降低高达20%方面超过了竞争基线，尽管根据攻击性的不同，GPT-2 Small的流畅性可能会显着下降。至关重要的是，转向后的标准NLP基准分数保持稳定，这表明模型的知识和一般能力得到了保留。我们进一步表明，更广泛的严重不良事件中的特征分裂会阻碍安全干预，强调了解开特征学习的重要性。我们的研究结果强调了LLM解毒基于CAE的因果干预措施的前景和当前的局限性，进一步为更安全的语言模型部署提出了实用指南。



## **3. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出非凡的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个关键问题出现了：当推理与危害交织在一起时，LRM是否会在推理模式中变得更容易越狱？为了研究这一点，我们引入了HauntAttack，这是一种新颖的通用黑匣子对抗攻击框架，它系统地将有害指令嵌入到推理问题中。具体来说，我们用有害指令修改现有问题中的关键推理条件，从而构建一条推理路径，引导模型逐步走向不安全的输出。我们对11种LRM进行了HauntAttack评估，观察到平均攻击成功率为70%，比之前最强的基线实现了高达12个百分点的绝对改进。我们的进一步分析表明，即使是先进的安全性一致的模型仍然极易受到基于推理的攻击，这为未来模型开发中平衡推理能力和安全性的紧迫挑战提供了见解。



## **4. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

超越文本：通过感性简单的转换对视觉语言和音频模型进行多模式越狱 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.

摘要: 多模式大型语言模型（MLLM）已经取得了显着的进展，但仍然极易受到利用跨模式处理弱点的对抗攻击的影响。我们对针对视觉语言和音频语言模型的多模式越狱进行了系统性研究，表明即使是简单的感知转换也可以可靠地绕过最先进的安全过滤器。我们的评估涵盖了三个高风险安全类别有害内容、CBRN（化学、生物、放射、核）和CTEM（儿童性剥削材料）的1，900个对抗性提示，针对七个前沿模型进行了测试。我们探索了MLLM攻击技术的有效性，包括FigStep-Pro（视觉关键字分解）、智能掩蔽（语义混淆）和音频扰动（Wave-Echo、Wave-Pitch、Wave-Speed）。结果揭示了严重的漏洞：在感知修改的输入下，具有几乎完美的纯文本安全性（0\%ASB）的模型遭受了超过75%的攻击成功率，而FigStep-Pro在Lama-4变体中实现了高达89%的ASB。基于音频的攻击进一步揭示了提供商特定的弱点，即使是基本的模式传输也会产生25%的技术查询的ASB。这些发现暴露了以文本为中心的对齐和多模式威胁之间的关键差距，表明当前的保障措施未能普遍适用于跨模式攻击。这些攻击的可访问性需要最少的技术专业知识，这表明强大的多模式人工智能安全性需要范式转向更广泛的语义层面推理，以减轻可能的风险。



## **5. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

TRUST：审计大型语言模型推理的去中心化框架 cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.

摘要: 大型语言模型生成复杂的推理链，揭示其决策，但验证这些中间步骤的忠实性和无害性仍然是一个尚未解决的关键问题。现有的审计方法集中、不透明且难以扩展，为在高风险领域部署专有模型带来了巨大风险。我们确定了四个核心挑战：（1）稳健性：集中式审计员是单点失败，容易受到偏见或攻击。(2)可扩展性：推理轨迹太长，无法手动验证。(3)不透明：封闭审计破坏了公众信任。(4)隐私：暴露完整推理可能会导致模型被盗或提炼。我们提出TRUST，这是一个透明的、去中心化的审计框架，通过以下方式克服这些限制：（1）不同审计员之间的共识机制，保证在高达30%的恶意参与者下的正确性。(2)推理痕迹的分层DAB分解，实现可扩展的并行审计。(3)一个区块链分类帐，记录所有验证决定，以供公众问责。(4)保留隐私的分段，仅共享部分推理步骤以保护专有逻辑。我们为TRUST框架的安全性和经济激励提供理论保证。跨多个LLM（GPT-OSS、DeepSeek-r1、Qwen）和推理任务（数学、医学、科学、人文学科）的实验表明，TRUST有效地检测推理缺陷，并在对抗性审计员的情况下保持稳健性。我们的工作开创了去中心化的人工智能审计，为安全且值得信赖的LLM部署提供了实用途径。



## **6. SAID: Empowering Large Language Models with Self-Activating Internal Defense**

SAID：通过自我激活的内部防御来增强大型语言模型 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20129v1) [paper-pdf](http://arxiv.org/pdf/2510.20129v1)

**Authors**: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen

**Abstract**: Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems.

摘要: 尽管在安全一致方面取得了进步，大型语言模型（LLM）仍然容易受到旨在绕过保护机制的越狱攻击。流行的防御策略依赖于外部干预，例如输入过滤或输出修改，这些干预通常缺乏可概括性并损害模型效用，同时产生大量的计算费用。在这项工作中，我们引入了一种新的免训练防御范式--自我激活内部防御（SAID），它将防御任务从外部纠正重新构建为内部能力激活。SAID独特地利用LLM自身的推理能力，通过三阶段管道主动识别和抵消恶意意图：模型原生意图提炼以提取核心语义，最佳安全前置探测以激活潜在安全意识，以及保守的聚合策略以确保稳健的决策。针对六种高级越狱攻击对五种开源LLM进行的广泛实验表明，SAID在减少有害输出方面远远优于最先进的防御。至关重要的是，它在实现这一目标的同时保留了良性任务的模型性能并产生最小的计算负担。我们的工作确定，激活LLM的本质安全机制是构建更安全、更可靠的一致人工智能系统的一条更稳健、更可扩展的途径。



## **7. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **8. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

Lexo：通过LLM辅助程序再生消除隐形供应链攻击 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.14522v2) [paper-pdf](http://arxiv.org/pdf/2510.14522v2)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.

摘要: 软件供应链攻击是开源软件生态系统中一个重要且持续存在的问题。这些攻击保留了组件实现的标准功能，但还隐藏了仅在组件到达其目标环境时激活的恶意功能。Lexo通过自动学习和重新生成潜在恶意组件的无可识别性版本来解决此类隐形攻击。Lexo首先生成一组输入-输出对来建模组件的完整可观察行为，然后使用其合成原始组件的新版本。新组件实现了原始功能，但避免了隐蔽的恶意行为。在整个重建过程中，Lexo会咨询大型语言模型（LLM）的几个不同实例，使用正确性和覆盖率指标来引导这些实例，并保护它们的结果。我们对100多个现实世界的包（包括高调的隐形供应链攻击）的评估表明，Lexo可以跨多个域扩展，有效地再生代码（平均<100），保持兼容性，并成功消除了几个现实世界的供应链攻击中的恶意代码，即使在最先进的LLM在提示时未能消除恶意代码的情况下也是如此。



## **9. SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical Tensors for Large Language Model Deployment**

SecureInfer：用于大型语言模型部署的隐私关键张量的异类TEE-Ginger架构 cs.CR

Accepted at IEEE Intelligent Computing and Systems at the Edge  (ICEdge) 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19979v1) [paper-pdf](http://arxiv.org/pdf/2510.19979v1)

**Authors**: Tushar Nayan, Ziqi Zhang, Ruimin Sun

**Abstract**: With the increasing deployment of Large Language Models (LLMs) on mobile and edge platforms, securing them against model extraction attacks has become a pressing concern. However, protecting model privacy without sacrificing the performance benefits of untrusted AI accelerators, such as GPUs, presents a challenging trade-off. In this paper, we initiate the study of high-performance execution on LLMs and present SecureInfer, a hybrid framework that leverages a heterogeneous Trusted Execution Environments (TEEs)-GPU architecture to isolate privacy-critical components while offloading compute-intensive operations to untrusted accelerators. Building upon an outsourcing scheme, SecureInfer adopts an information-theoretic and threat-informed partitioning strategy: security-sensitive components, including non-linear layers, projection of attention head, FNN transformations, and LoRA adapters, are executed inside an SGX enclave, while other linear operations (matrix multiplication) are performed on the GPU after encryption and are securely restored within the enclave. We implement a prototype of SecureInfer using the LLaMA-2 model and evaluate it across performance and security metrics. Our results show that SecureInfer offers strong security guarantees with reasonable performance, offering a practical solution for secure on-device model inference.

摘要: 随着大型语言模型（LLM）在移动和边缘平台上的部署越来越多，保护它们免受模型提取攻击已成为一个紧迫的问题。然而，在不牺牲不受信任的人工智能加速器（例如图形处理器）的性能优势的情况下保护模型隐私，提出了一个具有挑战性的权衡。在本文中，我们启动了LLM上高性能执行的研究，并提出了SecureInfer，这是一个混合框架，利用异类可信执行环境（TEEs）-图形处理器架构来隔离隐私关键组件，同时将计算密集型操作卸载到不受信任的加速器。SecureInfer在外包方案的基础上采用了信息理论和威胁知情的分区策略：安全敏感组件，包括非线性层、注意力投射、FNN变换和LoRA适配器，在SGX飞地内执行，而其他线性操作（矩阵相乘）在加密后在图形处理器上执行，并在飞地内安全地恢复。我们使用LLaMA-2模型实现SecureInfer的原型，并跨性能和安全指标对其进行评估。我们的结果表明，SecureInfer提供了强大的安全保证和合理的性能，为安全的设备上模型推断提供了实用的解决方案。



## **10. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

未学习但未被遗忘：LLM中精确未学习后的数据提取 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。



## **11. The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability Without Reference Models**

The Tail Tells All：Estimating Model-Level Membership Inference Vulnerability Without Reference Models（没有参考模型的模型级成员关系推断漏洞估计） cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19773v1) [paper-pdf](http://arxiv.org/pdf/2510.19773v1)

**Authors**: Euodia Dodd, Nataša Krčo, Igor Shilov, Yves-Alexandre de Montjoye

**Abstract**: Membership inference attacks (MIAs) have emerged as the standard tool for evaluating the privacy risks of AI models. However, state-of-the-art attacks require training numerous, often computationally expensive, reference models, limiting their practicality. We present a novel approach for estimating model-level vulnerability, the TPR at low FPR, to membership inference attacks without requiring reference models. Empirical analysis shows loss distributions to be asymmetric and heavy-tailed and suggests that most points at risk from MIAs have moved from the tail (high-loss region) to the head (low-loss region) of the distribution after training. We leverage this insight to propose a method to estimate model-level vulnerability from the training and testing distribution alone: using the absence of outliers from the high-loss region as a predictor of the risk. We evaluate our method, the TNR of a simple loss attack, across a wide range of architectures and datasets and show it to accurately estimate model-level vulnerability to the SOTA MIA attack (LiRA). We also show our method to outperform both low-cost (few reference models) attacks such as RMIA and other measures of distribution difference. We finally evaluate the use of non-linear functions to evaluate risk and show the approach to be promising to evaluate the risk in large-language models.

摘要: 成员资格推理攻击（MIA）已成为评估人工智能模型隐私风险的标准工具。然而，最先进的攻击需要训练大量且计算成本高昂的参考模型，从而限制了它们的实用性。我们提出了一种新的方法，用于估计模型级脆弱性（低FPR时的TPA），而不需要参考模型。经验分析表明损失分布是不对称的和重尾的，并表明MIA的大多数风险点在训练后已从分布的尾部（高损失区域）转移到了头部（低损失区域）。我们利用这一见解提出了一种仅根据训练和测试分布来估计模型级脆弱性的方法：使用高损失区域中不存在异常值作为风险的预测因子。我们在广泛的架构和数据集上评估了我们的方法（简单损失攻击的TNR），并展示它可以准确估计SOTA MIA攻击（LiRA）的模型级漏洞。我们还展示了我们的方法，可以优于RMIA等低成本（少数参考模型）攻击和其他分布差异测量。我们最终评估了使用非线性函数来评估风险，并展示了有希望在大型语言模型中评估风险的方法。



## **12. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

你能相信你所看到的吗？Alpha通道对视频对象检测的无框攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.

摘要: 随着对象检测模型越来越多地部署在自动驾驶汽车（AV）和监控平台等网络物理系统中，确保其针对对抗威胁的安全性至关重要。虽然之前的工作探讨了图像领域中的对抗攻击，但视频领域中的这些攻击在很大程度上仍然没有得到审查，尤其是在无框环境中。在本文中，我们介绍了{\Alpha}-Cloak，这是对对象检测器的第一个无箱对抗攻击，完全通过RGBA视频的Alpha通道进行操作。{\Alpha}-Cloak利用Alpha通道将恶意目标视频与良性视频融合，导致融合后的视频对人类观看者来说似乎无害，但始终欺骗对象检测器。我们的攻击不需要访问模型架构、参数或输出，并且不会引入可感知的伪影。我们系统性地研究了对常见视频格式和播放应用程序中Alpha通道的支持，并设计了一种融合算法来确保视觉隐形性和兼容性。我们在五个最先进的对象检测器、视觉语言模型和多模式大型语言模型（Gemini-2.0-Flash）上评估了{\Alpha}-Cloak，展示了在所有场景下100%的攻击成功率。我们的研究结果揭示了基于视频的感知系统中以前未探索的漏洞，凸显了对对抗环境中阿尔法通道的防御的迫切需要。



## **13. Machine Text Detectors are Membership Inference Attacks**

机器文本检测器是会员推断攻击 cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19492v1) [paper-pdf](http://arxiv.org/pdf/2510.19492v1)

**Authors**: Ryuto Koike, Liam Dugan, Masahiro Kaneko, Chris Callison-Burch, Naoaki Okazaki

**Abstract**: Although membership inference attacks (MIAs) and machine-generated text detection target different goals, identifying training samples and synthetic texts, their methods often exploit similar signals based on a language model's probability distribution. Despite this shared methodological foundation, the two tasks have been independently studied, which may lead to conclusions that overlook stronger methods and valuable insights developed in the other task. In this work, we theoretically and empirically investigate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. For our theoretical contribution, we prove that the metric that achieves the asymptotically highest performance on both tasks is the same. We unify a large proportion of the existing literature in the context of this optimal metric and hypothesize that the accuracy with which a given method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments, including 7 state-of-the-art MIA methods and 5 state-of-the-art machine text detectors across 13 domains and 10 generators, demonstrate very strong rank correlation (rho > 0.6) in cross-task performance. We notably find that Binoculars, originally designed for machine text detection, achieves state-of-the-art performance on MIA benchmarks as well, demonstrating the practical impact of the transferability. Our findings highlight the need for greater cross-task awareness and collaboration between the two research communities. To facilitate cross-task developments and fair evaluations, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, with implementation of 15 recent methods from both tasks.

摘要: 尽管成员资格推理攻击（MIA）和机器生成的文本检测针对不同的目标，识别训练样本和合成文本，但它们的方法通常基于语言模型的概率分布利用相似的信号。尽管有这一共同的方法论基础，但这两项任务是独立研究的，这可能会得出的结论忽视了另一项任务中开发的更强的方法和有价值的见解。在这项工作中，我们从理论和经验上研究了可转让性，即，最初为一项任务开发的方法在MIA和机器文本检测之间在另一项任务上的表现如何。对于我们的理论贡献，我们证明在两项任务中实现渐进最高性能的指标是相同的。我们在这个最佳指标的背景下统一了大部分现有文献，并假设给定方法逼近这个指标的准确性与其可移植性直接相关。我们的大规模实证实验，包括7种最先进的MIA方法和5种最先进的机器文本检测器，跨越13个域和10个生成器，证明了跨任务性能中非常强的等级相关性（rho > 0.6）。我们特别发现，最初为机器文本检测而设计的双筒望远镜在MIA基准测试上也实现了最先进的性能，证明了可移植性的实际影响。我们的研究结果强调了两个研究团体之间需要更大的跨任务意识和合作。为了促进跨任务开发和公平评估，我们引入了MINT，这是一个用于MIA和机器生成文本检测的统一评估套件，并从这两个任务中实现了15种最新方法。



## **14. Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation**

通过节点评估监控基于LLM的多代理系统防止损坏 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19420v1) [paper-pdf](http://arxiv.org/pdf/2510.19420v1)

**Authors**: Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, Meng Sun

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems.

摘要: 基于大语言模型（LLM）的多智能体系统（MAS）已成为人工智能应用的流行范式。然而，MAS的可信度问题仍然是一个关键问题。与单代理系统中的挑战不同，MAS涉及更复杂的通信过程，使其容易受到腐败攻击。为了缓解这个问题，基于MAS的图表示开发了多种防御机制，其中代理代表节点，通信形成边。然而，这些方法主要关注静态图防御，试图检测固定图结构中的攻击或优化具有某些防御能力的静态布局。为了解决这一局限性，我们提出了一种针对MAS图结构的动态防御范式，该范式持续监控MAS图内的通信，然后动态调整图布局，准确干扰恶意通信，并有效防御不断发展和多样化的动态攻击。在日益复杂和动态的MAS环境中的实验结果表明，我们的方法显着优于现有的MAS防御机制，为其值得信赖的应用提供了有效的护栏。我们的代码可在https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems上获取。



## **15. Defending Against Prompt Injection with DataFilter**

使用数据过滤器防御提示注入 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.

摘要: 当大型语言模型（LLM）代理越来越多地被部署来自动化任务并与不受信任的外部数据交互时，即时注入成为一个重大的安全威胁。通过将恶意指令注入LLM访问的数据中，攻击者可以任意覆盖原始用户任务，并将代理重定向到无意的、可能有害的操作。现有的防御要么需要访问模型权重（微调），导致大量效用损失（基于检测），要么要求进行非平凡的系统重新设计（系统级）。出于此动机，我们提出了数据过滤器，这是一种测试时模型不可知的防御，可以在数据到达后台LLM之前从数据中删除恶意指令。数据过滤器通过模拟注入的监督微调进行训练，并利用用户的指令和数据来选择性地剥离对抗内容，同时保留良性信息。在多个基准测试中，数据过滤器一致地将即时注入攻击成功率降低到接近零，同时保持LLM的实用性。数据过滤器提供强大的安全性、高实用性和即插即用部署，使其成为保护黑匣子商业LLM免受即时注入的强大实用防御。我们的数据过滤器模型已在https://huggingface.co/JoyYizhu/DataFilter上发布，可立即使用，其代码可在https://github.com/yizhu-joy/DataFilter上重现我们的结果。



## **16. OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform**

OpenGuardrails：一个开源上下文感知人工智能Guardrails平台 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19169v1) [paper-pdf](http://arxiv.org/pdf/2510.19169v1)

**Authors**: Thomas Wang, Haowen Li

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use.

摘要: 随着大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，保护它们免受不安全、恶意或侵犯隐私的内容的侵害至关重要。我们介绍了OpenGuardrails，这是第一个提供上下文感知安全和操纵检测模型以及全面人工智能护栏可部署平台的开源项目。OpenGuardrails可防止内容安全风险、模型操纵攻击（例如，提示注入、越狱、代码解释器滥用以及恶意代码的生成/执行）以及数据泄露。内容安全和模型操纵检测由统一的大模型实现，而数据泄露识别和编辑由单独的轻量级NER管道执行（例如，Presidio风格模型或基于regex的检测器）。该系统可以部署为安全网关或基于API的服务，并具有企业级的完全私有部署选项。OpenGuardrails在安全基准方面实现了最先进的（SOTA）性能，在英语、中文和多语言任务中的提示和响应分类方面都表现出色。所有模型均在Apache 2.0许可下发布，供公众使用。



## **17. PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits**

PLAGUE：终身自适应多回合漏洞生成的即插即用框架 cs.CR

First two authors have equal author contributions

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.17947v2) [paper-pdf](http://arxiv.org/pdf/2510.17947v2)

**Authors**: Neeladri Bhuiya, Madhav Aggarwal, Diptanshu Purwar

**Abstract**: Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation.

摘要: 大型语言模型（LLM）正在以惊人的速度改进。随着代理工作流程的出现，多轮对话已成为与LLM互动的事实模式，以完成漫长而复杂的任务。虽然LLM能力不断提高，但它们仍然越来越容易受到越狱的影响，特别是在多回合场景中，有害意图可能会巧妙地注入到整个对话中以产生邪恶结果。虽然单回合攻击已得到广泛探索，但适应性、效率和有效性仍然是多回合攻击的关键挑战。为了解决这些差距，我们提出了PLAGUE，这是一个新颖的即插即用框架，用于设计受终身学习代理启发的多回合攻击。PLAGUE将多回合攻击的生命周期分解为三个精心设计的阶段（Primer、Planner和Timisher），以便对多回合攻击家族进行系统性且信息丰富的探索。评估表明，使用PLAGUE设计的红色团队代理实现了最先进的越狱结果，以较少或相当的查询预算将领先模型的攻击成功率（ASB）提高了30%以上。特别是，PLAGUE在OpenAI的o3上的ASB（基于Strongestival）为81.4%，在Claude的Opus 4.1上为67.3%，这两种模型在安全文献中被认为对越狱具有高度抵抗力。我们的工作提供了工具和见解，以了解计划初始化、上下文优化和终身学习在制定多回合攻击以进行全面模型漏洞评估方面的重要性。



## **18. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

NEXUS：在多轮LLM越狱中利用不安全序列的网络探索 cs.CR

This paper has been accepted in the main conference proceedings of  the 2025 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.03417v2) [paper-pdf](http://arxiv.org/pdf/2510.03417v2)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但仍然容易受到越狱攻击，特别是在良性交换中散布恶意意图并绕过对齐机制的多回合越狱。现有的方法常常无法很好地探索对抗空间，依赖于手工制作的启发式方法，或者缺乏系统性的查询细化。我们介绍了NEXUS（用于eXploiting Unsafe Sequences的网络探索），这是一个用于构建、细化和执行优化多回合攻击的模块化框架。NEXUS包括：（1）IncreghtNet，它将有害意图分层扩展到主题、实体和查询链的结构化语义网络中;（2）反馈驱动的模拟器，通过攻击者-受害者-法官LLM协作使用危害性和语义相似性基准来迭代细化和修剪这些链;（3）网络穿越器，自适应地导航细化查询空间以进行实时攻击。该管道揭示了LLC之间隐秘、高成功的对抗路径。在几种闭源和开源LLM上，NEXUS将攻击成功率比之前的方法提高了2.1%至19.4%。代码：https://github.com/inspire-lab/NEXUS



## **19. HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models**

HarmNet：对大型语言模型进行自适应多回合越狱攻击的框架 cs.CR

This paper has been accepted for presentation at the Conference on  Applied Machine Learning in Information Security (CAMLIS 2025)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18728v1) [paper-pdf](http://arxiv.org/pdf/2510.18728v1)

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement.

摘要: 大型语言模型（LLM）仍然容易受到多回合越狱攻击。我们引入了HarmNet，这是一个模块化框架，包括分层语义网络InghtNet;用于迭代查询细化的反馈驱动模拟器;以及用于实时自适应攻击执行的Network Traverser。HarmNet系统性地探索和完善对抗空间，以发现隐秘、高成功的攻击路径。跨闭源和开源LLM的实验表明，HarmNet优于最先进的方法，实现了更高的攻击成功率。例如，在Mistral-7 B上，HarmNet的攻击成功率达到了99.4%，比最佳基线高出13.9%。索引术语：越狱攻击;大型语言模型;对抗性框架;查询细化。



## **20. Exploring Membership Inference Vulnerabilities in Clinical Large Language Models**

探索临床大型语言模型中的隶属推理漏洞 cs.CR

Accepted at the 1st IEEE Workshop on Healthcare and Medical Device  Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18674v1) [paper-pdf](http://arxiv.org/pdf/2510.18674v1)

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems.

摘要: 随着大型语言模型（LLM）越来越嵌入临床决策支持、文档和患者信息系统中，确保其隐私和可信度已成为医疗保健行业的一个紧迫挑战。对敏感电子健康记录（EHR）数据进行微调LLM可以改善域对齐，但也会增加通过模型行为暴露患者信息的风险。在这项正在进行的工作中，我们对临床LLM中的隶属关系推断漏洞进行了一项探索性实证研究，重点关注对手是否可以推断模型训练期间是否使用了特定的患者记录。使用最先进的临床问答模型Llemr，我们评估了典型的基于损失的攻击和更真实地反映临床对抗状况的基于领域动机的基于重述的扰动策略。我们的初步研究结果揭示了有限但可测量的会员泄露，这表明当前的临床LLM提供了部分抵抗力，但仍然容易受到微妙的隐私风险，这可能会破坏人们对临床人工智能采用的信任。这些结果激励了上下文感知、特定领域隐私评估和防御的持续发展，例如差异隐私微调和转述感知训练，以加强医疗保健人工智能系统的安全性和可信度。



## **21. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection**

SentinelNet：通过基于信用的动态威胁检测保护多代理协作 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16219v2) [paper-pdf](http://arxiv.org/pdf/2510.16219v2)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

摘要: 恶意代理对大型语言模型（LLM）支持的多代理系统（MAS）的可靠性和决策能力构成重大威胁。由于反应式设计或集中式架构可能会引入单点故障，现有的防御往往会出现缺陷。为了应对这些挑战，我们提出了SentinelNet，这是第一个用于主动检测和缓解多代理协作中恶意行为的去中心化框架。SentinelNet为每个代理配备了一个基于信用的检测器，该检测器通过对增强的对抗辩论轨迹进行对比学习进行训练，从而能够通过底部k消除来自主评估消息可信度和动态邻居排名，以抑制恶意通信。为了克服攻击数据的稀缺性，它生成模拟不同威胁的对抗轨迹，确保稳健的训练。MAS基准测试的实验表明，SentinelNet实现了对恶意代理近乎完美的检测，在两轮辩论中接近100%，并从受损的基线恢复了95%的系统准确性。SentinelNet在跨域和攻击模式之间表现出强大的概括性，建立了一种用于保护协作MAS的新颖范式。



## **22. The Attribution Story of WhisperGate: An Academic Perspective**

WhisperGate的归因故事：学术视角 cs.CR

Virus Bulletin Conference 2025

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18484v1) [paper-pdf](http://arxiv.org/pdf/2510.18484v1)

**Authors**: Oleksandr Adamov, Anders Carlsson

**Abstract**: This paper explores the challenges of cyberattack attribution, specifically APTs, applying the case study approach for the WhisperGate cyber operation of January 2022 executed by the Russian military intelligence service (GRU) and targeting Ukrainian government entities. The study provides a detailed review of the threat actor identifiers and taxonomies used by leading cybersecurity vendors, focusing on the evolving attribution from Microsoft, ESET, and CrowdStrike researchers. Once the attribution to Ember Bear (GRU Unit 29155) is established through technical and intelligence reports, we use both traditional machine learning classifiers and a large language model (ChatGPT) to analyze the indicators of compromise (IoCs), tactics, and techniques to statistically and semantically attribute the WhisperGate attack. Our findings reveal overlapping indicators with the Sandworm group (GRU Unit 74455) but also strong evidence pointing to Ember Bear, especially when the LLM is fine-tuned or contextually augmented with additional intelligence. Thus, showing how AI/GenAI with proper fine-tuning are capable of solving the attribution challenge.

摘要: 本文采用俄罗斯军事情报部门（GRU）2022年1月执行的针对乌克兰政府实体的WhisperGate网络行动的案例研究方法，探讨了网络攻击归因（特别是APT）的挑战。该研究详细回顾了领先网络安全供应商使用的威胁行为者标识符和分类法，重点关注Microsoft、ESET和CrowdStrike研究人员不断变化的归因。一旦通过技术和情报报告确定了Ember Bear（GRU Unit 29155）的归因，我们就使用传统的机器学习分类器和大型语言模型（ChatGPT）来分析妥协指标（IoCs）、策略和技术，以统计和语义上对WhisperGate攻击进行归因。我们的研究结果揭示了与Sandworm小组（GRU Unit 74455）重叠的指标，但也有强有力的证据指向Ember Bear，特别是当LLM经过微调或根据上下文使用额外的情报增强时。因此，展示了经过适当微调的AI/GenAI如何能够解决归因挑战。



## **23. DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning**

DeepTX：通过多模式特征和LLM推理进行实时交易风险分析 cs.CR

Accepted to ASE'25

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18438v1) [paper-pdf](http://arxiv.org/pdf/2510.18438v1)

**Authors**: Yixuan Liu, Xinlei Li, Yi Li

**Abstract**: Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).

摘要: Web 3生态系统中的网络钓鱼攻击越来越复杂，利用欺骗性合同逻辑、恶意前端脚本和代币批准模式。我们介绍了DeepTX，这是一个实时交易分析系统，可以在用户确认之前检测此类威胁。DeepTX模拟未决事务，提取行为、上下文和UI功能，并使用多个大型语言模型（LLM）来推理事务意图。具有自我反思的共识机制确保了稳健且可解释的决策。经过我们的网络钓鱼数据集的评估，DeepTX实现了高精度和召回率（演示视频：https：//youtu.be/4OfK9KCEXUM）。



## **24. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models**

SoK：大型语言模型中提示安全性的分类和评估 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.15476v2) [paper-pdf](http://arxiv.org/pdf/2510.15476v2)

**Authors**: Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing Huang, Yuan Hong

**Abstract**: Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date;\footnote{The dataset is released at \href{https://huggingface.co/datasets/youbin2014/JailbreakDB}{\textcolor{purple}{https://huggingface.co/datasets/youbin2014/JailbreakDB}}.} and (5) presenting a comprehensive evaluation platform and leaderboard of state-of-the-art methods \footnote{will be released soon.}. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment.

摘要: 大型语言模型（LLM）已迅速成为现实世界应用程序的组成部分，为不同领域的服务提供动力。然而，它们的广泛部署暴露了严重的安全风险，特别是通过越狱提示，这些提示可能绕过模型对齐并引发有害输出。尽管对攻击和防御技术进行了深入的研究，但该领域仍然支离破碎：定义、威胁模型和评估标准差异很大，阻碍了系统性进步和公平比较。在知识系统化（SoK）中，我们通过以下方式解决这些挑战：（1）提出一种整体、多级别的分类法，组织LLM即时安全中的攻击、防御和漏洞;（2）将威胁模型和成本假设形式化为机器可读配置文件，以进行可重复的评估;（3）引入开源评估工具包，用于标准化、可审计的攻击和防御比较;（4）发布JAILBREAKDB，这是迄今为止最大的越狱和良性提示注释数据集;\脚注{该数据集发布于\href{https：//huggingface.co/juets/youbin2014/JailbreakDB}{\textColor{purple}{https：//huggingface.co/juets/youbin2014/JailbreakDB}。}以及（5）提供全面的评估平台和最先进方法排行榜\脚注{将很快发布。}。我们的工作统一了碎片化的研究，为未来的研究提供了严格的基础，并支持开发适合高风险部署的稳健、值得信赖的LLM。



## **25. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟儿抓住了漏洞：在LLM服务系统中揭开定时侧通道 cs.CR

This work was first submitted for review on Sept. 5, 2024, and the  initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version  has accepted for publication by IEEE Transactions on Information Forensics  and Security (TIFS)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2409.20002v5) [paper-pdf](http://arxiv.org/pdf/2409.20002v5)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **26. Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming**

Genesis：LLM Web Agent Red-Teaming不断发展的攻击策略 cs.AI

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18314v1) [paper-pdf](http://arxiv.org/pdf/2510.18314v1)

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.

摘要: 随着大型语言模型（LLM）代理越来越多地自动化复杂的Web任务，它们提高了生产力，同时引入了新的安全风险。然而，关于Web代理攻击的相关研究仍然有限。现有的红色团队方法主要依赖于手动设计的攻击策略或离线训练的静态模型。此类方法无法捕捉Web代理的底层行为模式，因此很难在不同的环境中进行概括。在Web代理攻击中，成功需要不断发现和进化攻击策略。为此，我们提出了Genesis，这是一个由三个模块组成的新颖的代理框架：攻击者、得分者和策略者。攻击者通过将遗传算法与混合策略表示集成来生成对抗注入。评分者评估目标Web代理的响应以提供反馈。策略师从交互日志中动态发现有效的策略，并将其汇编到不断增长的策略库中，然后重新部署该库以增强攻击者的有效性。跨各种Web任务的广泛实验表明，我们的框架发现了新颖的策略，并且始终优于现有的攻击基线。



## **27. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

通过上下文空间对计算机使用代理进行安全有效的访问控制 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2509.22256v2) [paper-pdf](http://arxiv.org/pdf/2509.22256v2)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.36% of attacks while introducing only 6.83% performance overhead.

摘要: 基于大型语言模型（LLM）的计算机使用代理代表了人工智能和操作系统功能的融合，使自然语言能够控制系统和应用程序级功能。然而，由于LLM固有的不确定性问题，授予代理对计算机的控制权会带来巨大的安全风险。当代理行为偏离用户意图时，可能会导致不可逆转的后果。现有的缓解方法，例如用户确认和基于LLM的动态动作验证，仍然受到可用性、安全性和性能方面的限制。为了解决这些挑战，我们提出了CSAgent，这是一个用于计算机使用代理的系统级、基于静态策略的访问控制框架。为了弥合静态策略与动态上下文和用户意图之间的差距，CSAgent引入了意图和上下文感知策略，并提供了自动化工具链来帮助开发人员构建和完善它们。CSAgent通过优化的操作系统服务执行这些策略，确保代理操作只能在特定的用户意图和上下文下执行。CSAgent支持保护通过各种接口（包括API、CLI和图形用户界面）控制计算机的代理。我们实施并评估CSAgent，它成功防御了超过99.36%的攻击，同时仅引入了6.83%的性能负载。



## **28. DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents**

DrunkAgent：LLM-Powered Recommender Agent中的隐形内存损坏 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2503.23804v3) [paper-pdf](http://arxiv.org/pdf/2503.23804v3)

**Authors**: Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.   To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.

摘要: 大型语言模型（LLM）驱动的代理越来越多地用于推荐系统（RS）中以实现个性化行为建模，其中记忆机制在使代理能够自主探索、学习和从现实世界的交互中自我进化方面发挥着关键作用。然而，作为上下文存储库的这种机制本质上暴露了潜在对抗操纵的攻击表面。尽管代理RS发挥着核心作用，但面对此类威胁时的稳健性在很大程度上仍然没有得到充分的探索。之前的作品存在语义不匹配或依赖于静态嵌入或预定义的提示，所有这些都不是为动态系统设计的，尤其是为LLM代理的动态内存状态。商业收件箱的黑匣子性质加剧了这一挑战。   为了解决上述问题，在本文中，我们对LLM支持的推荐代理中基于内存的漏洞进行了首次系统性调查，揭示了它们的安全局限性，并指导加强系统弹性和可信度的努力。具体来说，我们提出了一种新颖的黑匣子攻击框架DrunkAgent。DrunkAgent为目标物品促销精心设计了具有语义意义的对抗性文本触发器，并引入了一系列策略，通过破坏交互期间的记忆更新来最大化触发效应。触发器和策略在代理模型上进行了优化，使DrunkAgent具有可转移性和隐蔽性。跨不同代理RS对现实世界数据集进行的广泛实验，包括协作过滤、检索增强和顺序推荐，证明了DrunkAgent的可概括性、可移植性和隐蔽性。



## **29. Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth**

任意深度对齐：解锁LLM到任意深度的固有安全对齐 cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18081v1) [paper-pdf](http://arxiv.org/pdf/2510.18081v1)

**Authors**: Jiawei Zhang, Andrew Estornell, David D. Baek, Bo Li, Xiaojun Xu

**Abstract**: Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial).

摘要: 大型语言模型（LLM）表现出强而浅的一致性：当在助理转向开始时预计会拒绝时，它们会直接拒绝有害查询，但一旦有害的延续正在进行（无论是通过对抗性攻击还是通过有害的助理预填充攻击），这种保护就会崩溃。这提出了一个基本问题：LLM中固有的浅对齐能否被解锁，以确保任意世代深度的安全性？为了实现这一目标，我们提出了任意深度对齐（ADA），这是一种有效的推断时防御，且费用可以忽略不计。ADA是基于我们的观察而构建的，即通过在浅层拒绝训练中的重复使用，对齐集中在辅助头部代币中，并且这些代币拥有模型的强大对齐先验。通过在中途重新引入这些代币，ADA引导模型重新评估危害性并在生成过程中的任何时刻恢复拒绝。在各种开源模型系列（Llama、Gemma、Mistral、Qwen、DeepSeek和gtt-oss）中，ADA实现了强大的安全性能，无需对基本模型的参数进行任何更改。它确保了近100%的拒绝率，以应对数十到数千个代币的具有挑战性的对抗性预填充攻击。此外，ADA还将突出的对抗性提示攻击（例如GCG、AutoDAN、PAIR和RAP）的平均成功率降低至3%以下。这一切都是在保持良性任务的实用性的同时实现的，同时尽量减少过度拒绝。即使在基本模型经历后续指令调整（良性或对抗）之后，ADA也会保持这种弹性。



## **30. CourtGuard: A Local, Multiagent Prompt Injection Classifier**

CourtGuard：本地、多代理即时注入分类器 cs.CR

11 pages, 7 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.19844v1) [paper-pdf](http://arxiv.org/pdf/2510.19844v1)

**Authors**: Isaac Wu, Michael Maslowski

**Abstract**: As large language models (LLMs) become integrated into various sensitive applications, prompt injection, the use of prompting to induce harmful behaviors from LLMs, poses an ever increasing risk. Prompt injection attacks can cause LLMs to leak sensitive data, spread misinformation, and exhibit harmful behaviors. To defend against these attacks, we propose CourtGuard, a locally-runnable, multiagent prompt injection classifier. In it, prompts are evaluated in a court-like multiagent LLM system, where a "defense attorney" model argues the prompt is benign, a "prosecution attorney" model argues the prompt is a prompt injection, and a "judge" model gives the final classification. CourtGuard has a lower false positive rate than the Direct Detector, an LLM as-a-judge. However, CourtGuard is generally a worse prompt injection detector. Nevertheless, this lower false positive rate highlights the importance of considering both adversarial and benign scenarios for the classification of a prompt. Additionally, the relative performance of CourtGuard in comparison to other prompt injection classifiers advances the use of multiagent systems as a defense against prompt injection attacks. The implementations of CourtGuard and the Direct Detector with full prompts for Gemma-3-12b-it, Llama-3.3-8B, and Phi-4-mini-instruct are available at https://github.com/isaacwu2000/CourtGuard.

摘要: 随着大型语言模型（LLM）集成到各种敏感应用程序中，提示注入（使用提示来诱导LLM的有害行为）构成了越来越大的风险。即时注入攻击可能导致LLM泄露敏感数据、传播错误信息并表现出有害行为。为了抵御这些攻击，我们提出了CourtGuard，这是一种可本地运行的多代理提示注入分类器。在其中，提示在类似法庭的多代理LLM系统中进行评估，其中“辩护律师”模型认为提示是良性的，“检察律师”模型认为提示是提示注射，“法官”模型给出最终分类。CourtGuard的假阳性率低于Direct Detector（LLM作为法官）。然而，CourtGuard通常是一个更糟糕的提示注射检测器。然而，这种较低的假阳性率凸显了在提示分类时考虑敌对和良性场景的重要性。此外，CourtGuard与其他即时注入分类器相比的相对性能促进了多智能体系统作为防御即时注入攻击的使用。CourtGuard和Direct Detector的实现（Gemma-3- 12 b-it、Llama-3.3-8B和Phi-4-mini-directory）可在https://github.com/isaacwu2000/CourtGuard上获取。



## **31. Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks**

人类恶意在特工中的回响：针对多轮在线骚扰攻击对标LLM cs.AI

13 pages, 4 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.14207v2) [paper-pdf](http://arxiv.org/pdf/2510.14207v2)

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.

摘要: 大型语言模型（LLM）代理正在为越来越多的交互式Web应用程序提供支持，但仍然容易受到滥用和伤害。之前的越狱研究主要集中在单回合提示上，而真正的骚扰通常在多回合互动中展开。在这项工作中，我们提出了在线骚扰统计基准，包括：（i）合成的多回合骚扰对话数据集，（ii）多代理（例如，骚扰者、受害者）由重复博弈理论指导的模拟，（iii）跨越记忆、规划和微调攻击代理的三种越狱方法，以及（iv）混合方法评估框架。我们利用两个突出的LLM，LLaMA-3.1-8B-Instruct（开源）和Gemini-2.0-flash（闭源）。我们的研究结果表明，越狱调整使骚扰几乎可以保证，在Llama中，攻击成功率为95.78- 96.89% vs. 57.25- 64.19%，在Gemini中，攻击成功率为99.33% vs. 98.46%，同时在两种模型中，拒绝率都大幅降低到1-2%。最普遍的有毒行为是侮辱，84.9- 87.8%对44.2- 50.8%，没有调整，81.2- 85.1%对31.5- 38.8%，表明与敏感类别相比，如性或种族骚扰，护栏较弱。定性评估进一步表明，攻击代理再现人类一样的侵略配置文件，如马基雅维利/心理变态模式下的计划，和自恋倾向的记忆。与直觉相反，闭源模型和开源模型在不同时期表现出不同的升级轨迹，而闭源模型则表现出显着的脆弱性。总体而言，我们的研究结果表明，多回合和基于理论的攻击不仅能够高成功率，而且还模拟了类人的骚扰动态，推动了强大的安全护栏的开发，以最终确保在线平台的安全和负责任。



## **32. Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution**

多语言LLM水印真的是多语言的吗？简单的反翻译解决方案 cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18019v1) [paper-pdf](http://arxiv.org/pdf/2510.18019v1)

**Authors**: Asim Mohamed, Martin Gubri

**Abstract**: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages.

摘要: 多语言水印的目标是使大语言模型（LLM）输出跨语言可追踪，但目前的方法仍然存在不足。尽管声称跨语言的鲁棒性，但它们只在高资源语言上进行评估。我们发现，现有的多语言水印方法是不是真正的多语言：他们未能保持稳健的翻译攻击下，在中等和低资源的语言。我们将这种失败追溯到语义聚类，当标记器词汇表包含的给定语言的全词标记太少时，它会失败。为了解决这个问题，我们引入了STEAM，这是一种基于反向翻译的检测方法，可以恢复通过翻译丢失的水印强度。STEAM与任何水印方法兼容，在不同的标记器和语言中稳健，非侵入性，并且可以轻松扩展到新语言。STEAM在17种语言上的平均收益为+0.19 AUC和+40%p TPA @1%，提供了一种简单而稳健的方法，以实现跨不同语言的更公平的水印。



## **33. VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models**

VERA-V：越狱视觉语言模型的变分推理框架 cs.CR

18 pages, 7 Figures,

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17759v1) [paper-pdf](http://arxiv.org/pdf/2510.17759v1)

**Authors**: Qilin Liao, Anamika Lochab, Ruqi Zhang

**Abstract**: Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o.

摘要: 视觉语言模型（VLM）通过视觉推理扩展了大型语言模型，但它们的多模式设计也引入了新的、未充分探索的漏洞。现有的多模式红色团队方法很大程度上依赖于脆弱的模板，专注于单一攻击设置，并且仅暴露漏洞的一小部分。为了解决这些限制，我们引入了VERA-V，这是一个变分推理框架，它将多模式越狱发现重新构建为学习成对文本图像提示上的联合后验分布。这种概率观点使得能够生成绕过模型护栏的隐形、耦合的对抗输入。我们训练轻量级攻击者来逼近后验，从而能够对各种越狱进行有效抽样，并提供对漏洞的分布见解。VERA-V进一步集成了三种补充策略：（i）嵌入有害线索的基于印刷术的文本提示，（ii）引入对抗信号的基于扩散的图像合成，以及（iii）碎片VLM注意力的结构化干扰物。HarmBench和HADES基准测试的实验表明，VERA-V在开源和前沿VLM上的表现始终优于最先进的基线，比GPT-4 o上的最佳基线高出53.75%。



## **34. CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks**

CrossGuard：保护MLLM免受联合模式隐性恶意攻击 cs.CR

14 pages, 8 figures, 2 tables

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17687v1) [paper-pdf](http://arxiv.org/pdf/2510.17687v1)

**Authors**: Xu Zhang, Hao Li, Zhichao Lu

**Abstract**: Multimodal Large Language Models (MLLMs) achieve strong reasoning and perception capabilities but are increasingly vulnerable to jailbreak attacks. While existing work focuses on explicit attacks, where malicious content resides in a single modality, recent studies reveal implicit attacks, in which benign text and image inputs jointly express unsafe intent. Such joint-modal threats are difficult to detect and remain underexplored, largely due to the scarcity of high-quality implicit data. We propose ImpForge, an automated red-teaming pipeline that leverages reinforcement learning with tailored reward modules to generate diverse implicit samples across 14 domains. Building on this dataset, we further develop CrossGuard, an intent-aware safeguard providing robust and comprehensive defense against both explicit and implicit threats. Extensive experiments across safe and unsafe benchmarks, implicit and explicit attacks, and multiple out-of-domain settings demonstrate that CrossGuard significantly outperforms existing defenses, including advanced MLLMs and guardrails, achieving stronger security while maintaining high utility. This offers a balanced and practical solution for enhancing MLLM robustness against real-world multimodal threats.

摘要: 多模式大型语言模型（MLLM）实现了强大的推理和感知能力，但越来越容易受到越狱攻击。虽然现有的工作重点是显式攻击，其中恶意内容存在于单一模式中，但最近的研究揭示了隐式攻击，其中良性文本和图像输入共同表达不安全意图。此类联合模式威胁很难检测到并且仍然未充分探索，这主要是由于缺乏高质量的隐性数据。我们提出ImpForge，这是一个自动化的红色团队管道，它利用强化学习和定制的奖励模块来生成跨14个领域的多样化隐式样本。在此数据集的基础上，我们进一步开发了CrossGuard，这是一种意图感知的保护措施，可针对显式和隐性威胁提供强大而全面的防御。跨安全和不安全基准、隐式和显式攻击以及多种域外设置的广泛实验表明，CrossGuard的性能显着优于现有防御（包括高级MLLM和护栏），在保持高实用性的同时实现了更强的安全性。这提供了一个平衡且实用的解决方案，可以增强MLLM针对现实世界的多模式威胁的稳健性。



## **35. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which  was submitted as a new work by accident

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2508.09201v2) [paper-pdf](http://arxiv.org/pdf/2508.09201v2)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致更高的AUROC检测。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **36. Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs**

观看权重：对微调LLM进行无监督监控和控制 cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2508.00161v2) [paper-pdf](http://arxiv.org/pdf/2508.00161v2)

**Authors**: Ziqian Zhong, Aditi Raghunathan

**Abstract**: The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.   In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.   For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation.   Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.

摘要: 强大的开权大型语言模型（LLM）的发布通常不会伴随着对其完整训练数据的访问。现有的可解释性方法，尤其是基于激活的方法，通常需要或假设分布相似的数据。在检测和防御后门等新型潜在威胁时，这是一个重大限制，根据定义，后门是无法分发的。   在这项工作中，我们引入了一种用于理解、监控和控制微调LLM的新方法，该方法解释权重而不是激活，从而减少了对分布上与未知训练数据相似的数据的需求。我们证明，微调模型与其基本模型之间权重差的顶级奇异载体对应于新获得的行为。通过监控沿着这些方向的激活的cos相似性，我们可以高精度地检测微调期间引入的显着行为。   对于在存在秘密触发器时绕过安全机制的后门模型，我们的方法可以阻止高达100%的攻击，误报率低于1.2%。对于经历了未学习的模型，我们检测对已删除主题的推断，准确率高达95.42%，甚至可以引导模型恢复“未学习”的信息。除了监控之外，我们的方法还显示了部署前模型审计的潜力：通过分析商业化的预调模型（OLMo，Llama，Qwen），我们能够发现特定于模型的微调重点，包括营销策略和中途提示生成。   我们的实现可以在https://github.com/fjzzq2002/WeightWatch上找到。



## **37. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs**

OCR-APT：使用子图异常检测和LLM从审计工作组重建APT故事 cs.CR

This is the authors' extended version of the paper accepted for  publication at the ACM SIGSAC Conference on Computer and Communications  Security (CCS 2025). The final published version is available at  https://doi.org/10.1145/3719027.3765219

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.15188v2) [paper-pdf](http://arxiv.org/pdf/2510.15188v2)

**Authors**: Ahmed Aly, Essam Mansour, Amr Youssef

**Abstract**: Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.

摘要: 高级持续威胁（APT）是一种隐秘的网络攻击，通常可以逃避系统级审计日志中的检测。起源图将这些日志建模为相连的实体和事件，揭示线性日志表示错过的关系。现有的系统将异常检测应用于这些图形，但通常存在高假阳性率和粗粒度警报的问题。它们对文件路径或IP等节点属性的依赖会导致虚假相关性，从而降低检测稳健性和可靠性。为了充分了解攻击的进展和影响，安全分析师需要能够对整个攻击生成准确、类似人类的叙述的系统。为了应对这些挑战，我们引入了OCR-APT，这是一种用于APT检测和重建类人攻击故事的系统。OCR-APT使用图神经网络（GNN）进行子图异常检测，学习节点周围的行为模式，而不是文件路径或IP等脆弱属性。这种方法可以实现更强大的异常检测。然后，它使用大型语言模型（LLM）对检测到的子图进行迭代，以重建多阶段攻击故事。每个阶段都在进行之前进行验证，减少幻觉并确保可解释的最终报告。我们对DARPA TC 3、OpTC和NODLINK数据集的评估表明，OCR-APT在检测准确性和警报可解释性方面都优于最先进的系统。此外，OCR-APT重建了全面捕捉攻击故事的人性化报告。



## **38. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CV

Withdrawn due to an accidental duplicate submission. This paper  (arXiv:2510.15430) was unintentionally submitted as a new entry instead of a  new version of our previous work (arXiv:2508.09201)

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.15430v2) [paper-pdf](http://arxiv.org/pdf/2510.15430v2)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致更高的AUROC检测。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **39. Agentic Reinforcement Learning for Search is Unsafe**

搜索的强化学习是不安全的 cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17431v1) [paper-pdf](http://arxiv.org/pdf/2510.17431v1)

**Authors**: Yushi Yang, Shreyansh Padarha, Andrew Lee, Adam Mahdi

**Abstract**: Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search.

摘要: 抽象强化学习（RL）训练大型语言模型在推理期间自主调用工具，其中搜索是最常见的应用。这些模型擅长多步推理任务，但其安全属性尚未得到很好的理解。在这项研究中，我们表明RL训练的搜索模型继承了指令调优的拒绝，并且经常通过将有害请求转化为安全查询来转移它们。然而，这种安全性是脆弱的。两种简单的攻击，一种迫使模型开始通过搜索进行响应（搜索攻击），另一种鼓励模型重复搜索（多搜索攻击），引发了有害搜索和答案的级联。在两个同时具有本地和网络搜索的模式家族（Qwen、Llama）中，这些攻击将拒绝率降低了高达60.0%，答案安全性降低了82.5%，搜索查询安全性降低了82.4%。这些攻击通过在模型生成继承的拒绝令牌之前触发模型生成有害的请求镜像搜索查询来成功。这暴露了当前RL训练的一个核心弱点：它奖励持续生成有效查询，而不考虑其危害性。因此，RL搜索模型存在用户可以轻松利用的漏洞，因此迫切需要开发安全意识的代理RL管道来优化安全搜索。



## **40. Semantic Representation Attack against Aligned Large Language Models**

针对对齐大型语言模型的语义表示攻击 cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2509.19360v2) [paper-pdf](http://arxiv.org/pdf/2509.19360v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.   Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.   We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.   Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.   This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.   The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.   We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.   Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.   The code will be publicly available.

摘要: 大型语言模型（LLM）越来越多地使用对齐技术来防止有害输出。尽管有这些保护措施，攻击者仍然可以通过制作提示来诱导LLM生成有害内容来规避它们。   当前的方法通常针对确切的肯定回应，例如“当然，这里是..”，遭受收敛有限、提示不自然和计算成本高的困扰。   我们引入了语义表示攻击，这是一种新颖的范式，它从根本上重新概念化了针对一致的LLM的对抗目标。   我们的方法不是针对精确的文本模式，而是利用了由具有等效有害含义的不同响应组成的语义表示空间。   这项创新解决了困扰现有方法的攻击功效和即时自然性之间固有的权衡。   提出了语义表示启发式搜索算法，通过在增量扩展期间保持可解释性，有效地生成语义一致和简洁的对抗提示。   我们为语义融合建立了严格的理论保证，并证明我们的方法实现了前所未有的攻击成功率（18个LLM平均为89.41%，其中11个模型为100%），同时保持隐蔽性和效率。   全面的实验结果证实了我们的语义表示攻击的整体优势。   该代码将公开。



## **41. Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses**

文本属性图学习的鲁棒性：见解、权衡和新防御 cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17185v1) [paper-pdf](http://arxiv.org/pdf/2510.17185v1)

**Authors**: Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, Zhewei Wei, Yongchao Liu, Chuntao Hong

**Abstract**: While Graph Neural Networks (GNNs) and Large Language Models (LLMs) are powerful approaches for learning on Text-Attributed Graphs (TAGs), a comprehensive understanding of their robustness remains elusive. Current evaluations are fragmented, failing to systematically investigate the distinct effects of textual and structural perturbations across diverse models and attack scenarios. To address these limitations, we introduce a unified and comprehensive framework to evaluate robustness in TAG learning. Our framework evaluates classical GNNs, robust GNNs (RGNNs), and GraphLLMs across ten datasets from four domains, under diverse text-based, structure-based, and hybrid perturbations in both poisoning and evasion scenarios. Our extensive analysis reveals multiple findings, among which three are particularly noteworthy: 1) models have inherent robustness trade-offs between text and structure, 2) the performance of GNNs and RGNNs depends heavily on the text encoder and attack type, and 3) GraphLLMs are particularly vulnerable to training data corruption. To overcome the identified trade-offs, we introduce SFT-auto, a novel framework that delivers superior and balanced robustness against both textual and structural attacks within a single model. Our work establishes a foundation for future research on TAG security and offers practical solutions for robust TAG learning in adversarial environments. Our code is available at: https://github.com/Leirunlin/TGRB.

摘要: 虽然图神经网络（GNN）和大型语言模型（LLM）是在文本属性图（TAG）上学习的强大方法，但对其稳健性的全面理解仍然难以捉摸。当前的评估是支离破碎的，未能系统地调查文本和结构扰动对不同模型和攻击场景的独特影响。为了解决这些限制，我们引入了一个统一且全面的框架来评估TAG学习的稳健性。我们的框架在中毒和逃避场景中的各种基于文本、基于结构和混合的扰动下，评估了来自四个领域的十个数据集的经典GNN、稳健GNN（RGNN）和GraphLLM。我们的广泛分析揭示了多个发现，其中三个特别值得注意：1）模型在文本和结构之间具有固有的鲁棒性权衡，2）GNN和RGNN的性能严重取决于文本编码器和攻击类型，3）GraphLLM特别容易受到训练数据损坏的影响。为了克服已确定的权衡，我们引入了SFT-Auto，这是一种新颖的框架，可以在单个模型内针对文本和结构性攻击提供卓越且平衡的鲁棒性。我们的工作为未来TAG安全性研究奠定了基础，并为对抗环境中稳健的TAG学习提供了实用的解决方案。我们的代码可访问：https://github.com/Leirunlin/TGRB。



## **42. ARMOR: Aligning Secure and Safe Large Language Models via Meticulous Reasoning**

ARMOR：通过量化推理调整安全且安全的大型语言模型 cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2507.11500v2) [paper-pdf](http://arxiv.org/pdf/2507.11500v2)

**Authors**: Zhengyue Zhao, Yingzi Ma, Somesh Jha, Marco Pavone, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Models have shown impressive generative capabilities across diverse tasks, but their safety remains a critical concern. Existing post-training alignment methods, such as SFT and RLHF, reduce harmful outputs yet leave LLMs vulnerable to jailbreak attacks, especially advanced optimization-based ones. Recent system-2 approaches enhance safety by adding inference-time reasoning, where models assess potential risks before producing responses. However, we find these methods fail against powerful out-of-distribution jailbreaks, such as AutoDAN-Turbo and Adversarial Reasoning, which conceal malicious goals behind seemingly benign prompts. We observe that all jailbreaks ultimately aim to embed a core malicious intent, suggesting that extracting this intent is key to defense. To this end, we propose ARMOR, which introduces a structured three-step reasoning pipeline: (1) analyze jailbreak strategies from an external, updatable strategy library, (2) extract the core intent, and (3) apply policy-based safety verification. We further develop ARMOR-Think, which decouples safety reasoning from general reasoning to improve both robustness and utility. Evaluations on advanced optimization-based jailbreaks and safety benchmarks show that ARMOR achieves state-of-the-art safety performance, with an average harmful rate of 0.002 and an attack success rate of 0.06 against advanced optimization-based jailbreaks, far below other reasoning-based models. Moreover, ARMOR demonstrates strong generalization to unseen jailbreak strategies, reducing their success rate to zero. These highlight ARMOR's effectiveness in defending against OOD jailbreak attacks, offering a practical path toward secure and reliable LLMs.

摘要: 大型语言模型在不同任务中表现出令人印象深刻的生成能力，但它们的安全性仍然是一个关键问题。现有的训练后对齐方法（例如SFT和WLHF）可以减少有害输出，但使LLM容易受到越狱攻击，尤其是基于高级优化的攻击。最近的system-2方法通过添加推理时推理来增强安全性，其中模型在做出响应之前评估潜在风险。然而，我们发现这些方法无法应对强大的分发外越狱，例如AutoDAN-Turbo和对抗推理，这些方法在看似良性的提示背后隐藏了恶意目标。我们观察到，所有越狱的最终目的都是嵌入核心恶意意图，这表明提取这一意图是防御的关键。为此，我们提出了ARMOR，它引入了结构化的三步推理管道：（1）从外部可更新策略库中分析越狱策略，（2）提取核心意图，（3）应用基于策略的安全验证。我们进一步开发了ARMOR-Think，它将安全推理与一般推理分开，以提高稳健性和实用性。对基于高级优化的越狱和安全基准的评估表明，ARMOR实现了最先进的安全性能，针对基于高级优化的越狱，平均有害率为0.002，攻击成功率为0.06，远低于其他基于推理的模型。此外，ARMOR对看不见的越狱策略表现出了很强的概括性，将其成功率降至零。这些凸显了ARMOR在防御OOD越狱攻击方面的有效性，为实现安全可靠的LLM提供了实用途径。



## **43. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

Video-SafetyBench：视频LVLM安全评估的基准 cs.CV

Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2505.11842v2) [paper-pdf](http://arxiv.org/pdf/2505.11842v2)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.

摘要: 大型视觉语言模型（LVLM）的越来越多的部署引发了潜在恶意输入下的安全问题。然而，现有的多模式安全评估主要关注静态图像输入暴露的模型漏洞，而忽略了可能引发明显安全风险的视频的时间动态。为了弥合这一差距，我们引入了Video-SafetyBench，这是第一个旨在评估LVLM在视频文本攻击下的安全性的综合基准。它由2，264个视频-文本对组成，涵盖48个细粒度的不安全类别，每个将合成视频与包含明显恶意的有害查询或良性查询配对，良性查询看似无害，但在与视频一起解释时会触发有害行为。为了生成语义准确的视频以进行安全评估，我们设计了一个可控的管道，将视频语义分解为主题图像（显示的内容）和运动文本（它如何移动），共同指导查询相关视频的合成。为了有效地评估不确定或边缘有害输出，我们提出了RJScore，这是一种新型的基于LLM的指标，它结合了判断模型的置信度和人类一致的决策阈值校准。大量实验表明，良性查询视频合成的平均攻击成功率为67.2%，揭示了视频诱导攻击的一致漏洞。我们相信Video-SafetyBench将促进未来对基于视频的安全评估和防御策略的研究。



## **44. Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models**

Transformer内存会被损坏吗？调查大型语言模型中的缓存端漏洞 cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17098v1) [paper-pdf](http://arxiv.org/pdf/2510.17098v1)

**Authors**: Elias Hossain, Swayamjit Saha, Somshubhra Roy, Ravi Prasad

**Abstract**: Even when prompts and parameters are secured, transformer language models remain vulnerable because their key-value (KV) cache during inference constitutes an overlooked attack surface. This paper introduces Malicious Token Injection (MTI), a modular framework that systematically perturbs cached key vectors at selected layers and timesteps through controlled magnitude and frequency, using additive Gaussian noise, zeroing, and orthogonal rotations. A theoretical analysis quantifies how these perturbations propagate through attention, linking logit deviations to the Frobenius norm of corruption and softmax Lipschitz dynamics. Empirical results show that MTI significantly alters next-token distributions and downstream task performance across GPT-2 and LLaMA-2/7B, as well as destabilizes retrieval-augmented and agentic reasoning pipelines. These findings identify cache integrity as a critical yet underexplored vulnerability in current LLM deployments, positioning cache corruption as a reproducible and theoretically grounded threat model for future robustness and security research.

摘要: 即使提示和参数是安全的，Transformer语言模型仍然很脆弱，因为它们在推理过程中的键值（KV）缓存构成了一个被忽视的攻击面。本文介绍了恶意令牌注入（MTI），这是一个模块化框架，它通过控制幅度和频率，使用加性高斯噪声，调零和正交旋转，在选定的层和时间步上系统地扰动缓存的关键向量。理论分析量化了这些扰动如何通过注意力传播，将logit偏差与腐败的Frobenius范数和softmax Lipschitz动力学联系起来。实证结果表明，MTI显着改变了GPT-2和LLaMA-2/7 B的下一个令牌分布和下游任务性能，以及不稳定的检索增强和代理推理管道。这些发现将缓存完整性确定为当前LLM部署中一个关键但未充分探索的漏洞，将缓存损坏定位为未来稳健性和安全性研究的可重复且理论依据的威胁模型。



## **45. System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection**

系统提示中毒：对大型语言模型的持续攻击超出用户注入 cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2505.06493v3) [paper-pdf](http://arxiv.org/pdf/2505.06493v3)

**Authors**: Zongze Li, Jiawei Guo, Haipeng Cai

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning.

摘要: 大型语言模型（LLM）因其令人印象深刻的生成能力而在不同的应用程序中得到广泛采用。其即插即用性质使开发人员和最终用户能够通过简单的提示与这些模型进行交互。然而，随着LLM越来越多地集成到不同领域的各种系统中，对其安全性的担忧也越来越大。现有的研究主要关注用户提示（例如提示注入攻击）和模型输出（例如模型倒置攻击）引起的威胁，而系统提示的安全性在很大程度上仍然被忽视。这项工作弥合了关键差距。我们引入了系统提示中毒，这是一种针对LLM的新攻击载体，与传统的用户提示注入不同，系统提示中毒，因此持续影响所有后续用户交互和模型响应。我们系统地研究了各种中毒场景下的四种实用攻击策略。通过生成式和推理式LLM的演示，我们表明系统提示中毒在不需要越狱技术的情况下是高度可行的，并且在广泛的任务中有效，包括数学、编码、逻辑推理和自然语言处理。重要的是，我们的研究结果表明，即使用户提示采用思想链（CoT）等高级提示技术，攻击仍然有效。我们还表明，这些技术，包括CoT和检索增强生成（RAG），被证明可以有效地提高广泛任务中的LLM性能，但其有效性因系统即时中毒而显着削弱。



## **46. Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning**

忘记忘记：注意力下沉作为后备LLM遗忘的门户 cs.LG

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17021v1) [paper-pdf](http://arxiv.org/pdf/2510.17021v1)

**Authors**: Bingqi Shang, Yiwei Chen, Yihua Zhang, Bingquan Shen, Sijia Liu

**Abstract**: Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.

摘要: 大型语言模型（LLM）unlearning已经成为从预训练模型中删除不需要的数据，知识或行为，同时保留其通用性的关键机制。然而，随着开放权重LLM的兴起，我们不禁要问：遗忘过程本身是否可以被隐藏起来，在正常情况下看似成功，但当隐藏的触发器被激活时，它又会恢复到之前的遗忘行为？从经典的后门攻击中汲取灵感，将触发器嵌入到训练数据中以执行特定的行为，我们研究了后门遗忘，其中模型在干净的设置中按预期遗忘，但在触发器出现时恢复遗忘的知识。我们表明，设计这样的攻击提出了独特的挑战，取决于触发器放置在哪里以及如何加强后门训练。我们发现后门功效和注意力下沉现象之间存在着密切的联系，即，浅层输入代币在LLM中始终吸引了不成比例的关注。我们的分析表明，这些注意力下沉是后门消除的门户：将触发器放置在下沉位置并调整它们的注意力值显着增强后门持久性。大量的实验验证了这些发现，表明注意力下沉引导的后门取消学习可以在后门触发器存在的情况下可靠地恢复被遗忘的知识，而在没有触发器时，从通常未学习的模型中表现出不可想象的行为。代码可在https://github.com/OPTML-Group/Unlearn-Backdoor上获取。



## **47. Online Learning Defense against Iterative Jailbreak Attacks via Prompt Optimization**

通过即时优化防御迭代越狱攻击的在线学习防御 cs.CL

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17006v1) [paper-pdf](http://arxiv.org/pdf/2510.17006v1)

**Authors**: Masahiro Kaneko, Zeerak Talat, Timothy Baldwin

**Abstract**: Iterative jailbreak methods that repeatedly rewrite and input prompts into large language models (LLMs) to induce harmful outputs -- using the model's previous responses to guide each new iteration -- have been found to be a highly effective attack strategy. Despite being an effective attack strategy against LLMs and their safety mechanisms, existing defenses do not proactively disrupt this dynamic trial-and-error cycle. In this study, we propose a novel framework that dynamically updates its defense strategy through online learning in response to each new prompt from iterative jailbreak methods. Leveraging the distinctions between harmful jailbreak-generated prompts and typical harmless prompts, we introduce a reinforcement learning-based approach that optimizes prompts to ensure appropriate responses for harmless tasks while explicitly rejecting harmful prompts. Additionally, to curb overfitting to the narrow band of partial input rewrites explored during an attack, we introduce Past-Direction Gradient Damping (PDGD). Experiments conducted on three LLMs show that our approach significantly outperforms five existing defense methods against five iterative jailbreak methods. Moreover, our results indicate that our prompt optimization strategy simultaneously enhances response quality for harmless tasks.

摘要: 迭代越狱方法重复重写并输入到大型语言模型（LLM）中以引发有害输出--使用模型之前的响应来指导每次新迭代--已被发现是一种非常有效的攻击策略。尽管现有的防御措施是针对LLM及其安全机制的有效攻击策略，但并不能主动破坏这个动态的试错循环。在这项研究中，我们提出了一种新颖的框架，该框架通过在线学习动态更新其防御策略，以响应迭代越狱方法的每个新提示。利用越狱产生的有害提示和典型无害提示之间的区别，我们引入了一种基于强化学习的方法，该方法优化提示，以确保对无害任务做出适当的响应，同时明确拒绝有害提示。此外，为了抑制对攻击期间探索的部分输入重写的狭窄范围的过度适应，我们引入了过去方向梯度衰减（PDGDing）。在三种LLM上进行的实验表明，我们的方法在对抗五种迭代越狱方法时的性能显着优于五种现有的防御方法。此外，我们的结果表明，我们的即时优化策略同时提高了无害任务的响应质量。



## **48. Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs**

每次查询泄露的位：针对LLM的对抗性攻击的信息理论界限 cs.CR

NeurIPS 2025 (spotlight)

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17000v1) [paper-pdf](http://arxiv.org/pdf/2510.17000v1)

**Authors**: Masahiro Kaneko, Timothy Baldwin

**Abstract**: Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.

摘要: 恶意用户发起的威胁大型语言模型（LLM）安全性的对抗攻击可以被视为试图推断目标属性$T$，该属性在发出指令时未知，并且只有在观察到模型的回复后才变得已知。目标属性$T$的示例包括触发LLM有害响应或拒绝的二进制标志，以及通过取消学习删除的信息可以恢复的程度，两者都是通过对抗指令引发的。LLM揭示了一个\{observable Signal} $Z$，该$Z $可能会通过包含答案令牌、思维过程令牌或logit的响应泄露攻击提示。然而，泄露的信息规模仍然是轶事，这使得审计师没有原则性的指导，而辩护人则对透明度--风险权衡视而不见。我们用信息理论框架填补了这一空白，该框架计算可以安全披露多少信息，并使审计师能够衡量他们的方法与基本极限的接近程度。将观察值$Z$和目标属性$T$之间的互信息$I（Z;T）$视为每个查询的泄露位，我们表明实现错误$\varepð $至少需要$\log（1/\varepŸ）/I（Z;T）$查询，以逆泄露率线性扩展，并且仅以所需的准确度进行数学计算。因此，即使披露量适度增加，攻击成本也会从二次下降到对数。在系统提示泄露、越狱和重新学习攻击中对七个LLM进行的实验证实了这一理论：仅暴露答案令牌就需要大约一千个查询;添加日志将其减少到大约一百个;而揭示完整的思维过程将其减少到几十个。我们的结果为部署LLM时平衡透明度和安全性提供了第一个原则性标准。



## **49. BreakFun: Jailbreaking LLMs via Schema Exploitation**

BreakFun：通过模式利用越狱LLM cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17904v1) [paper-pdf](http://arxiv.org/pdf/2510.17904v1)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.

摘要: 大型语言模型（LLM）在处理结构化数据和遵守语法规则方面的熟练程度是推动其广泛采用的一种能力，但也使它们变得脆弱。在本文中，我们通过BreakFun研究了这个漏洞，BreakFun是一种越狱方法，它将LLM坚持结构化模式作为武器。BreakFun采用了一个由三部分组成的提示，将一个无辜的框架和一个思想链分散注意力与一个核心“特洛伊模式”相结合-一个精心制作的数据结构，迫使模型生成有害内容，利用LLM遵循结构和模式的强烈倾向。我们证明这个漏洞是高度可转移的，在JailbreakBench上的13个基础和专有模型上实现了89%的平均成功率，并在几个突出的模型上达到了100%的攻击成功率（ASR）。一项严格的消融研究证实，该特洛伊模式是攻击的主要原因。为了解决这个问题，我们引入了对抗性提示解构护栏，这是一种利用二级LLM来执行“文字转录”的防御--提取所有人类可读的文本以隔离和揭示用户真正的有害意图。我们的概念验证护栏展示了针对攻击的高功效，验证了针对欺骗性模式是一种可行的缓解策略。我们的工作探讨了法学硕士的核心优势如何转化为关键弱点，为构建更稳健一致的模型提供了新的视角。



## **50. Black-box Optimization of LLM Outputs by Asking for Directions**

通过询问方向进行LLM输出的黑匣子优化 cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.16794v1) [paper-pdf](http://arxiv.org/pdf/2510.16794v1)

**Authors**: Jie Zhang, Meng Ding, Yang Liu, Jue Hong, Florian Tramèr

**Abstract**: We present a novel approach for attacking black-box large language models (LLMs) by exploiting their ability to express confidence in natural language. Existing black-box attacks require either access to continuous model outputs like logits or confidence scores (which are rarely available in practice), or rely on proxy signals from other models. Instead, we demonstrate how to prompt LLMs to express their internal confidence in a way that is sufficiently calibrated to enable effective adversarial optimization. We apply our general method to three attack scenarios: adversarial examples for vision-LLMs, jailbreaks and prompt injections. Our attacks successfully generate malicious inputs against systems that only expose textual outputs, thereby dramatically expanding the attack surface for deployed LLMs. We further find that better and larger models exhibit superior calibration when expressing confidence, creating a concerning security paradox where model capability improvements directly enhance vulnerability. Our code is available at this [link](https://github.com/zj-jayzhang/black_box_llm_optimization).

摘要: 我们提出了一种新的方法来攻击黑盒大语言模型（LLM），利用他们的能力来表达自然语言的信心。现有的黑盒攻击要么需要访问连续的模型输出，如logits或置信度得分（实际上很少可用），要么依赖于其他模型的代理信号。相反，我们展示了如何促使LLM以一种充分校准的方式表达其内部信心，以实现有效的对抗优化。我们将我们的一般方法应用于三种攻击场景：针对视觉LLM的对抗性示例，越狱和即时注入。我们的攻击成功地针对仅公开文本输出的系统生成恶意输入，从而极大地扩大了已部署LLM的攻击面。我们进一步发现，更好、更大的模型在表达信心时表现出更好的校准，从而产生了一个令人担忧的安全悖论，即模型能力的改进直接增强了脆弱性。我们的代码可在此[链接]（https：//github.com/zj-jayzhang/black_box_llm_optimification）上获取。



