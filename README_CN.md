# Latest Adversarial Attack Papers
**update at 2025-12-08 15:37:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

揭露Pink Slime新闻：语言签名和针对LLM生成的威胁的稳健检测 cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.

摘要: 当地新闻格局是2800万美国人可靠信息的重要来源，但面临着来自Pink Slime Journalism的日益增长的威胁，Pink Slime Journalism是一种模仿合法当地报道的低质量自动生成文章。检测这些欺骗性文章需要对其语言、文体和词汇特征进行细粒度分析。在这项工作中，我们进行了一项全面的研究，以揭示Pink Slime内容的区别模式，并根据这些见解提出检测策略。除了传统的生成方法之外，我们还强调了一种新的对抗性载体：通过大型语言模型（LLM）进行修改。我们的研究结果表明，即使是消费者可访问的LLM也会显着破坏现有的检测系统，使其F1评分的性能降低高达40%。为了应对这一威胁，我们引入了一个强大的学习框架，专门设计用于抵抗基于LLM的对抗攻击并适应自动化粉红粘液新闻不断变化的格局，并表现出高达27%的改进。



## **2. A Practical Honeypot-Based Threat Intelligence Framework for Cyber Defence in the Cloud**

基于蜜罐的实用威胁情报框架，用于云中网络防御 cs.CR

6 pages

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.05321v1) [paper-pdf](https://arxiv.org/pdf/2512.05321v1)

**Authors**: Darren Malvern Chin, Bilal Isfaq, Simon Yusuf Enoch

**Abstract**: In cloud environments, conventional firewalls rely on predefined rules and manual configurations, limiting their ability to respond effectively to evolving or zero-day threats. As organizations increasingly adopt platforms such as Microsoft Azure, this static defense model exposes cloud assets to zero-day exploits, botnets, and advanced persistent threats. In this paper, we introduce an automated defense framework that leverages medium- to high-interaction honeypot telemetry to dynamically update firewall rules in real time. The framework integrates deception sensors (Cowrie), Azure-native automation tools (Monitor, Sentinel, Logic Apps), and MITRE ATT&CK-aligned detection within a closed-loop feedback mechanism. We developed a testbed to automatically observe adversary tactics, classify them using the MITRE ATT&CK framework, and mitigate network-level threats automatically with minimal human intervention.   To assess the framework's effectiveness, we defined and applied a set of attack- and defense-oriented security metrics. Building on existing adaptive defense strategies, our solution extends automated capabilities into cloud-native environments. The experimental results show an average Mean Time to Block of 0.86 seconds - significantly faster than benchmark systems - while accurately classifying over 12,000 SSH attempts across multiple MITRE ATT&CK tactics. These findings demonstrate that integrating deception telemetry with Azure-native automation reduces attacker dwell time, enhances SOC visibility, and provides a scalable, actionable defense model for modern cloud infrastructures.

摘要: 在云环境中，传统防火墙依赖于预定义的规则和手动配置，限制了其有效响应不断变化或零日威胁的能力。随着组织越来越多地采用Microsoft Azure等平台，这种静态防御模型使云资产暴露于零日攻击、僵尸网络和高级持续威胁之下。在本文中，我们引入了一个自动防御框架，该框架利用中到高交互蜜罐遥感来实时动态更新防火墙规则。该框架在闭环反馈机制中集成了欺骗传感器（Cowrie）、Azure原生自动化工具（Monitor、Sentinel、Logic Apps）以及MITRE ATA和CK对齐检测。我们开发了一个测试平台来自动观察对手的战术，使用MITRE ATT & CK框架对其进行分类，并以最少的人为干预自动缓解网络级威胁。   为了评估框架的有效性，我们定义并应用了一组面向攻击和防御的安全指标。我们的解决方案基于现有的自适应防御策略，将自动化功能扩展到云原生环境中。实验结果显示平均阻止时间为0.86秒，比基准系统快得多，同时可以准确地分类多种MITRE ATT & CK策略中的12，000多次SSH尝试。这些发现表明，将欺骗遥感与Azure原生自动化集成可以减少攻击者的驻留时间，增强SOC可见性，并为现代云基础设施提供可扩展、可操作的防御模型。



## **3. Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems**

Chameleon：用于多模式人工智能系统中基于扩展的视觉提示注入的自适应对抗代理 cs.AI

5 pages, 2 figures, IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04895v1) [paper-pdf](https://arxiv.org/pdf/2512.04895v1)

**Authors**: M Zeeshan, Saud Satti

**Abstract**: Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.

摘要: 多模式人工智能（AI）系统，特别是视觉语言模型（VLM），已成为从自主决策到自动文档处理等关键应用不可或缺的一部分。随着这些系统的扩展，它们严重依赖预处理管道来有效处理不同的输入。然而，这种对标准预处理操作（特别是图像缩减）的依赖会产生一个严重但经常被忽视的安全漏洞。虽然缩放算法旨在实现计算优化，但可以利用缩放算法来隐藏恶意视觉提示，这些提示对人类观察者来说是不可见的，但一旦被模型处理就变成了活动的语义指令。当前的对抗策略在很大程度上仍然是静态的，未能考虑到现代代理工作流程的动态性质。为了解决这一差距，我们提出了Chameleon，这是一种新颖的、自适应的对抗框架，旨在暴露和利用生产VLM中的扩展漏洞。与传统的静态攻击不同，Chameleon采用基于代理的迭代优化机制，该机制根据目标模型的实时反馈动态细化图像扰动。这使得该框架能够制作高度稳健的对抗性示例，这些示例能够经受住标准缩减操作以劫持下游执行的考验。我们根据Gemini 2.5 Flash模型评估Chameleon。我们的实验表明，Chameleon在不同的缩放因子下实现了84.5%的攻击成功率（ASB），显着优于平均仅为32.1%的静态基线攻击。此外，我们表明这些攻击有效地损害了代理管道，使多步骤任务中的决策准确性降低了45%以上。最后，我们讨论了这些漏洞的影响，并提出多规模一致性检查作为必要的防御机制。



## **4. SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security**

SoK：大型语言模型安全性的全面因果分析框架 cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04841v1) [paper-pdf](https://arxiv.org/pdf/2512.04841v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality.

摘要: 大型语言模型（LLM）表现出非凡的能力，但仍然容易受到越狱等敌对操纵的影响，其中精心设计的提示绕过了安全机制。了解此类漏洞背后的因果因素对于构建可靠的防御至关重要。   在这项工作中，我们引入了一个统一的因果关系分析框架，该框架系统地支持LLM中所有层面的因果关系调查，从代币层面、神经元层面和层层面干预到代表层面分析。该框架能够在各种基于偶然性的攻击和防御方法之间进行一致的实验和比较。伴随着这一实施，我们对疏忽驱动的越狱研究进行了首次全面调查，并对多个开放权重模型和安全关键基准（包括越狱、幻觉检测、后门识别和公平性评估）的框架进行了实证评估。我们的结果表明：（1）对因果关键部件进行有针对性的干预可以可靠地改变安全行为;（2）安全相关机制高度局部化（即，集中在早期到中层，只有1- 2%的神经元表现出因果影响）;以及（3）从我们的框架中提取的因果特征在多种威胁类型中实现了超过95%的检测准确率。   通过连接理论因果关系分析和实际模型安全性，我们的框架为LLM中基于因果关系的攻击、可解释性以及稳健的攻击检测和缓解的研究奠定了可重复的基础。代码可在https://github.com/Amadeuszhao/SOK_Casuality上获取。



## **5. Malicious Image Analysis via Vision-Language Segmentation Fusion: Detection, Element, and Location in One-shot**

通过视觉语言分割融合进行恶意图像分析：一次性检测、元素和定位 cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04599v1) [paper-pdf](https://arxiv.org/pdf/2512.04599v1)

**Authors**: Sheng Hang, Chaoxiang He, Hongsheng Hu, Hanqing Hu, Bin Benjamin Zhu, Shi-Feng Sun, Dawu Gu, Shuo Wang

**Abstract**: Detecting illicit visual content demands more than image-level NSFW flags; moderators must also know what objects make an image illegal and where those objects occur. We introduce a zero-shot pipeline that simultaneously (i) detects if an image contains harmful content, (ii) identifies each critical element involved, and (iii) localizes those elements with pixel-accurate masks - all in one pass. The system first applies foundation segmentation model (SAM) to generate candidate object masks and refines them into larger independent regions. Each region is scored for malicious relevance by a vision-language model using open-vocabulary prompts; these scores weight a fusion step that produces a consolidated malicious object map. An ensemble across multiple segmenters hardens the pipeline against adaptive attacks that target any single segmentation method. Evaluated on a newly-annotated 790-image dataset spanning drug, sexual, violent and extremist content, our method attains 85.8% element-level recall, 78.1% precision and a 92.1% segment-success rate - exceeding direct zero-shot VLM localization by 27.4% recall at comparable precision. Against PGD adversarial perturbations crafted to break SAM and VLM, our method's precision and recall decreased by no more than 10%, demonstrating high robustness against attacks. The full pipeline processes an image in seconds, plugs seamlessly into existing VLM workflows, and constitutes the first practical tool for fine-grained, explainable malicious-image moderation.

摘要: 检测非法视觉内容需要的不仅仅是图像级别的NSFW标志;版主还必须知道什么对象使图像非法以及这些对象出现的位置。我们引入了一个零拍摄流水线，同时（i）检测图像是否包含有害内容，（ii）识别所涉及的每个关键元素，以及（iii）使用像素精确的掩模定位这些元素-所有这些都在一次通过中。该系统首先应用基础分割模型（SAM）生成候选对象掩模，并将其细化为较大的独立区域。每个区域都通过视觉语言模型使用开放式词汇提示进行恶意相关性评分;这些评分加权融合步骤，产生合并的恶意对象地图。跨多个分割器的集成可以增强管道抵御针对任何单一分割方法的自适应攻击。在涵盖毒品、性、暴力和极端主义内容的新注释的790张图像数据集上进行评估，我们的方法获得了85.8%的元素级召回率、78.1%的准确率和92.1%的片段成功率--在相当的精确度下，超过了直接零镜头VLM定位27.4%的召回率。针对旨在破解Sam和VLM的PVD对抗性扰动，我们的方法的精确度和召回率下降不超过10%，展示了对攻击的高度鲁棒性。完整的管道可以在几秒钟内处理图像，无缝插入现有的VLM工作流程，并构成了第一个用于细粒度、可解释的恶意图像审核的实用工具。



## **6. Counterfeit Answers: Adversarial Forgery against OCR-Free Document Visual Question Answering**

假冒答案：对抗伪造针对无OCR文档视觉问题回答 cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04554v1) [paper-pdf](https://arxiv.org/pdf/2512.04554v1)

**Authors**: Marco Pintore, Maura Pintor, Dimosthenis Karatzas, Battista Biggio

**Abstract**: Document Visual Question Answering (DocVQA) enables end-to-end reasoning grounded on information present in a document input. While recent models have shown impressive capabilities, they remain vulnerable to adversarial attacks. In this work, we introduce a novel attack scenario that aims to forge document content in a visually imperceptible yet semantically targeted manner, allowing an adversary to induce specific or generally incorrect answers from a DocVQA model. We develop specialized attack algorithms that can produce adversarially forged documents tailored to different attackers' goals, ranging from targeted misinformation to systematic model failure scenarios. We demonstrate the effectiveness of our approach against two end-to-end state-of-the-art models: Pix2Struct, a vision-language transformer that jointly processes image and text through sequence-to-sequence modeling, and Donut, a transformer-based model that directly extracts text and answers questions from document images. Our findings highlight critical vulnerabilities in current DocVQA systems and call for the development of more robust defenses.

摘要: 文档视觉问题解答（DocVQA）支持基于文档输入中存在的信息的端到端推理。虽然最近的模型表现出了令人印象深刻的能力，但它们仍然容易受到对抗攻击。在这项工作中，我们引入了一种新颖的攻击场景，旨在以视觉上不可感知但具有语义针对性的方式伪造文档内容，允许对手从DocVQA模型中诱导特定或通常不正确的答案。我们开发专门的攻击算法，可以生成针对不同攻击者目标量身定制的对抗伪造文档，范围从有针对性的错误信息到系统性模型失败场景。我们针对两个端到端的最新模型展示了我们的方法的有效性：Pix2Struct，一个通过序列到序列建模联合处理图像和文本的视觉语言Transformer，和Donut，一个基于转换器的模型，直接从文档图像中提取文本并回答问题。我们的研究结果强调了当前DocVQA系统中的关键漏洞，并呼吁开发更强大的防御系统。



## **7. Adversarial Limits of Quantum Certification: When Eve Defeats Detection**

量子认证的对抗限制：当伊芙击败检测时 quant-ph

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04391v1) [paper-pdf](https://arxiv.org/pdf/2512.04391v1)

**Authors**: Davut Emre Tasar

**Abstract**: Security of quantum key distribution (QKD) relies on certifying that observed correlations arise from genuine quantum entanglement rather than eavesdropper manipulation. Theoretical security proofs assume idealized conditions, practical certification must contend with adaptive adversaries who optimize their attack strategies against detection systems. Established fundamental adversarial limits for quantum certification using Eve GAN, a generative adversarial network trained to produce classical correlations indistinguishable from quantum. Our central finding: when Eve interpolates her classical correlations with quantum data at mixing parameter, all tested detection methods achieve ROC AUC = 0.50, equivalent to random guessing. This means an eavesdropper needs only 5% classical admixture to completely evade detection. Critically, we discover that same distribution calibration a common practice in prior certification studies inflates detection performance by 44 percentage points compared to proper cross distribution evaluation, revealing a systematic flaw that may have led to overestimated security claims. Analysis of Popescu Rohrlich (PR Box) regime identifies a sharp phase transition at CHSH S = 2.05: below this value, no statistical method distinguishes classical from quantum correlations; above it, detection probability increases monotonically. Hardware validation on IBM Quantum demonstrates that Eve-GAN achieves CHSH = 2.736, remarkably exceeding real quantum hardware performance (CHSH = 2.691), illustrating that classical adversaries can outperform noisy quantum systems on standard certification metrics. These results have immediate implications for QKD security: adversaries maintaining 95% quantum fidelity evade all tested detection methods. We provide corrected methodology using cross-distribution calibration and recommend mandatory adversarial testing for quantum security claims.

摘要: 量子密钥分发（QKD）的安全性依赖于证明观察到的相关性来自真正的量子纠缠，而不是窃听者操纵。理论安全证明假设理想化的条件，实际认证必须与自适应对手抗衡，后者优化针对检测系统的攻击策略。使用Eve GAN建立了量子认证的基本对抗限制，Eve GAN是一个生成对抗网络，经过训练，可产生与量子无法区分的经典相关性。我们的中心发现：当Eve在混合参数下用量子数据插值她的经典相关性时，所有测试的检测方法都实现了ROC AUC = 0.50，相当于随机猜测。这意味着窃听者仅需要5%的经典混合物即可完全逃避检测。至关重要的是，我们发现，与适当的交叉分布评估相比，相同的分布校准（先前认证研究中的常见做法）使检测性能提高了44个百分点，揭示了可能导致高估安全声明的系统性缺陷。对Popescu Rohrlich（PR Box）状态的分析发现，CHSH S = 2.05处存在急剧的相转变：低于该值，没有统计方法将经典相关性与量子相关性区分开来;高于该值，检测概率单调增加。IBM Quantum上的硬件验证表明，Eve-GAN达到了CHSH = 2.736，显着超过了真实的量子硬件性能（CHSH = 2.691），说明经典对手在标准认证指标上可以优于有噪的量子系统。这些结果对QKD安全性产生了直接影响：保持95%量子保真度的对手规避了所有测试的检测方法。我们使用交叉分布校准提供纠正的方法，并建议对量子安全主张进行强制对抗测试。



## **8. One Detector Fits All: Robust and Adaptive Detection of Malicious Packages from PyPI to Enterprises**

一个检测器适合所有人：对从PyPI到企业的恶意包进行稳健且自适应的检测 cs.CR

Proceedings of the 2025 Annual Computer Security Applications Conference (ACSAC' 25), December 8-12, 2025, Honolulu, Hawaii, USA

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04338v1) [paper-pdf](https://arxiv.org/pdf/2512.04338v1)

**Authors**: Biagio Montaruli, Luca Compagna, Serena Elisa Ponta, Davide Balzarotti

**Abstract**: The rise of supply chain attacks via malicious Python packages demands robust detection solutions. Current approaches, however, overlook two critical challenges: robustness against adversarial source code transformations and adaptability to the varying false positive rate (FPR) requirements of different actors, from repository maintainers (requiring low FPR) to enterprise security teams (higher FPR tolerance).   We introduce a robust detector capable of seamless integration into both public repositories like PyPI and enterprise ecosystems. To ensure robustness, we propose a novel methodology for generating adversarial packages using fine-grained code obfuscation. Combining these with adversarial training (AT) enhances detector robustness by 2.5x. We comprehensively evaluate AT effectiveness by testing our detector against 122,398 packages collected daily from PyPI over 80 days, showing that AT needs careful application: it makes the detector more robust to obfuscations and allows finding 10% more obfuscated packages, but slightly decreases performance on non-obfuscated packages.   We demonstrate production adaptability of our detector via two case studies: (i) one for PyPI maintainers (tuned at 0.1% FPR) and (ii) one for enterprise teams (tuned at 10% FPR). In the former, we analyze 91,949 packages collected from PyPI over 37 days, achieving a daily detection rate of 2.48 malicious packages with only 2.18 false positives. In the latter, we analyze 1,596 packages adopted by a multinational software company, obtaining only 1.24 false positives daily. These results show that our detector can be seamlessly integrated into both public repositories like PyPI and enterprise ecosystems, ensuring a very low time budget of a few minutes to review the false positives.   Overall, we uncovered 346 malicious packages, now reported to the community.

摘要: 通过恶意Python包进行的供应链攻击的兴起需要强大的检测解决方案。然而，当前的方法忽视了两个关键挑战：针对对抗性源代码转换的鲁棒性以及对不同参与者（从存储库维护者（需要低FPR）到企业安全团队（更高FPR容忍度）的不同误报率（FPR）要求的适应性。   我们引入了一个强大的检测器，能够无缝集成到PyPI等公共存储库和企业生态系统中。为了确保稳健性，我们提出了一种使用细粒度代码模糊生成对抗包的新颖方法。将这些与对抗训练（AT）相结合将检测器稳健性提高了2.5倍。我们通过针对80天内每天从PyPI收集的122，398个包裹来测试我们的检测器来全面评估AT的有效性，表明AT需要仔细应用：它使检测器对混淆更加稳健，并允许发现多10%的混淆包，但在非混淆包上的性能略有下降。   我们通过两个案例研究展示了我们检测器的生产适应性：（i）一个针对PyPI维护者（调整为0.1% FPR）和（ii）一个针对企业团队（调整为10% FPR）。在前者中，我们分析了37天内从PyPI收集的91，949个包，每日检测率为2.48个恶意包，误报率仅为2.18个。在后者中，我们分析了一家跨国软件公司采用的1，596个包，每天仅获得1.24个假阳性。这些结果表明，我们的检测器可以无缝集成到PyPI等公共存储库和企业生态系统中，从而确保审查假阳性的时间预算非常低，只需几分钟。   总体而言，我们发现了346个恶意包，现已报告给社区。



## **9. Studying Various Activation Functions and Non-IID Data for Machine Learning Model Robustness**

研究各种激活函数和非IID数据以实现机器学习模型的鲁棒性 cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04264v1) [paper-pdf](https://arxiv.org/pdf/2512.04264v1)

**Authors**: Long Dang, Thushari Hapuarachchi, Kaiqi Xiong, Jing Lin

**Abstract**: Adversarial training is an effective method to improve the machine learning (ML) model robustness. Most existing studies typically consider the Rectified linear unit (ReLU) activation function and centralized training environments. In this paper, we study the ML model robustness using ten different activation functions through adversarial training in centralized environments and explore the ML model robustness in federal learning environments. In the centralized environment, we first propose an advanced adversarial training approach to improving the ML model robustness by incorporating model architecture change, soft labeling, simplified data augmentation, and varying learning rates. Then, we conduct extensive experiments on ten well-known activation functions in addition to ReLU to better understand how they impact the ML model robustness. Furthermore, we extend the proposed adversarial training approach to the federal learning environment, where both independent and identically distributed (IID) and non-IID data settings are considered. Our proposed centralized adversarial training approach achieves a natural and robust accuracy of 77.08% and 67.96%, respectively on CIFAR-10 against the fast gradient sign attacks. Experiments on ten activation functions reveal ReLU usually performs best. In the federated learning environment, however, the robust accuracy decreases significantly, especially on non-IID data. To address the significant performance drop in the non-IID data case, we introduce data sharing and achieve the natural and robust accuracy of 70.09% and 54.79%, respectively, surpassing the CalFAT algorithm, when 40% data sharing is used. That is, a proper percentage of data sharing can significantly improve the ML model robustness, which is useful to some real-world applications.

摘要: 对抗训练是提高机器学习（ML）模型鲁棒性的有效方法。大多数现有的研究通常考虑矫正线性单元（ReLU）激活功能和集中式培训环境。本文通过集中式环境中的对抗训练，使用十种不同的激活函数研究ML模型的鲁棒性，并探索联邦学习环境中ML模型的鲁棒性。在集中式环境中，我们首先提出了一种先进的对抗训练方法，通过结合模型架构更改、软标签、简化的数据增强和变化的学习率来提高ML模型的稳健性。然后，除了ReLU之外，我们还对十个著名的激活函数进行了广泛的实验，以更好地了解它们如何影响ML模型的稳健性。此外，我们将拟议的对抗训练方法扩展到联邦学习环境，其中考虑了独立和同分布（IID）和非IID数据设置。我们提出的集中式对抗训练方法在CIFAR-10上针对快速梯度符号攻击，分别实现了77.08%和67.96%的自然和稳健准确性。对十个激活函数的实验表明，ReLU通常表现最好。然而，在联邦学习环境中，鲁棒准确性显着下降，尤其是在非IID数据上。为了解决非IID数据情况下性能显着下降的问题，我们引入了数据共享，并分别实现了70.09%和54.79%的自然和稳健准确性，超过了使用40%数据共享时的CalLAT算法。也就是说，适当比例的数据共享可以显着提高ML模型的稳健性，这对一些现实世界的应用程序很有用。



## **10. Out-of-the-box: Black-box Causal Attacks on Object Detectors**

开箱即用：对对象检测器的黑匣子因果攻击 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03730v1) [paper-pdf](https://arxiv.org/pdf/2512.03730v1)

**Authors**: Melane Navaratnarajah, David A. Kelly, Hana Chockler

**Abstract**: Adversarial perturbations are a useful way to expose vulnerabilities in object detectors. Existing perturbation methods are frequently white-box and architecture specific. More importantly, while they are often successful, it is rarely clear why they work. Insights into the mechanism of this success would allow developers to understand and analyze these attacks, as well as fine-tune the model to prevent them. This paper presents BlackCAtt, a black-box algorithm and a tool, which uses minimal, causally sufficient pixel sets to construct explainable, imperceptible, reproducible, architecture-agnostic attacks on object detectors. BlackCAtt combines causal pixels with bounding boxes produced by object detectors to create adversarial attacks that lead to the loss, modification or addition of a bounding box. BlackCAtt works across different object detectors of different sizes and architectures, treating the detector as a black box. We compare the performance of BlackCAtt with other black-box attack methods and show that identification of causal pixels leads to more precisely targeted and less perceptible attacks. On the COCO test dataset, our approach is 2.7 times better than the baseline in removing a detection, 3.86 times better in changing a detection, and 5.75 times better in triggering new, spurious, detections. The attacks generated by BlackCAtt are very close to the original image, and hence imperceptible, demonstrating the power of causal pixels.

摘要: 对抗性扰动是暴露对象检测器漏洞的有用方法。现有的扰动方法通常是白盒和特定于架构的。更重要的是，虽然它们往往是成功的，但人们很少清楚它们为何有效。深入了解这一成功的机制将使开发人员能够理解和分析这些攻击，并微调模型以防止它们。本文介绍了BlackCAtt，这是一种黑匣子算法和工具，它使用最小的、因果关系充分的像素集来构建对对象检测器的可解释、不可感知、可复制、架构不可知的攻击。BlackCAtt将因果像素与对象检测器产生的边界框相结合，以创建导致边界框丢失、修改或添加的对抗攻击。BlackCAtt适用于不同尺寸和架构的不同对象检测器，将检测器视为黑匣子。我们将BlackCAtt的性能与其他黑匣子攻击方法进行了比较，并表明因果像素的识别会导致更精确的针对性和更难感知的攻击。在COCO测试数据集上，我们的方法在消除检测方面比基线好2.7倍，在改变检测方面比基线好3.86倍，在触发新的虚假检测方面好5.75倍。BlackCAtt生成的攻击非常接近原始图像，因此难以察觉，证明了因果像素的力量。



## **11. Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs**

上下文感知分层学习：实现更安全的LLM的两步范式 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03720v1) [paper-pdf](https://arxiv.org/pdf/2512.03720v1)

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.

摘要: 大型语言模型（LLM）已成为各种应用程序的强大工具。然而，他们的统一令牌处理范式在指令处理中引入了关键漏洞，特别是当暴露于对抗场景时。在这项工作中，我们识别并提出了一类新型漏洞，称为工具完成攻击（MCA），它利用函数调用机制来颠覆模型行为。为了评估LLM针对此类威胁的稳健性，我们引入了工具完成基准，这是一个全面的安全评估框架，它表明即使是最先进的模型仍然容易受到MCA的影响，并且攻击成功率高得惊人。为了解决这些漏洞，我们引入了上下文感知分层学习（CAHL），这是一种复杂的机制，可以动态平衡语义理解与特定角色的指令约束。CAHL利用不同指令段之间的上下文相关性来建立稳健的、上下文感知的指令层次结构。大量实验表明，CAHL显着增强了LLM针对传统攻击和拟议的MCA的鲁棒性，在零激发评估中表现出强大的概括能力，同时仍然保留了通用任务的模型性能。我们的代码可以在https://github.com/S2AILab/CAHL上找到。



## **12. A Descriptive Model for Modelling Attacker Decision-Making in Cyber-Deception**

网络欺骗中攻击者决策建模的描述性模型 cs.CR

24 Pages, 4 Tables

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03641v1) [paper-pdf](https://arxiv.org/pdf/2512.03641v1)

**Authors**: B. R. Turner, O. Guidetti, N. M. Karie, R. Ryan, Y. Yan

**Abstract**: Cyber-deception is an increasingly important defensive strategy, shaping adversarial decision making through controlled misinformation, uncertainty, and misdirection. Although game-theoretic, Bayesian, Markov decision process, and reinforcement learning models offer insight into deceptive interactions, they typically assume an attacker has already chosen to engage. Such approaches overlook cognitive and perceptual factors that influence an attacker's initial decision to engage or withdraw. This paper presents a descriptive model that incorporates the psychological and strategic elements shaping this decision. The model defines five components, belief (B), scepticism (S), deception fidelity (D), reconnaissance (R), and experience (E), which interact to capture how adversaries interpret deceptive cues and assess whether continued engagement is worthwhile. The framework provides a structured method for analysing engagement decisions in cyber-deception scenarios. A series of experiments has been designed to evaluate this model through Capture the Flag activities incorporating varying levels of deception, supported by behavioural and biometric observations. These experiments have not yet been conducted, and no experimental findings are presented in this paper. These experiments will combine behavioural observations with biometric indicators to produce a multidimensional view of adversarial responses. Findings will improve understanding of the factors influencing engagement decisions and refine the model's relevance to real-world cyber-deception settings. By addressing the gap in existing models that presume engagement, this work supports more cognitively realistic and strategically effective cyber-deception practices.

摘要: 网络欺骗是一种越来越重要的防御策略，通过受控的错误信息、不确定性和误导来塑造对抗性决策。尽管博弈论、Bayesian、Markov决策过程和强化学习模型提供了对欺骗性交互的见解，但它们通常假设攻击者已经选择参与。此类方法忽视了影响攻击者最初参与或退出决定的认知和感知因素。本文提出了一个描述性模型，其中融合了塑造这一决定的心理和战略因素。该模型定义了五个组成部分：信念（B）、怀疑论（S）、欺骗忠实度（D）、侦察（R）和经验（E），它们相互作用以捕捉对手如何解释欺骗性线索并评估继续参与是否值得。该框架提供了一种结构化方法来分析网络欺骗场景中的参与决策。设计了一系列实验来通过包含不同程度欺骗的夺旗活动来评估该模型，并由行为和生物识别观察支持。这些实验尚未进行，本文中也没有给出实验结果。这些实验将将行为观察与生物识别指标相结合，以产生对抗反应的多维视图。研究结果将提高对影响参与决策的因素的理解，并完善模型与现实世界网络欺骗环境的相关性。通过解决假设参与的现有模型中的差距，这项工作支持更加认知现实和战略有效的网络欺骗实践。



## **13. FeatureLens: A Highly Generalizable and Interpretable Framework for Detecting Adversarial Examples Based on Image Features**

EnterpriseLens：一个高度可概括和可解释的框架，用于基于图像特征检测对抗性示例 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03625v1) [paper-pdf](https://arxiv.org/pdf/2512.03625v1)

**Authors**: Zhigang Yang, Yuan Liu, Jiawei Zhang, Puning Zhang, Xinqiang Ma

**Abstract**: Although the remarkable performance of deep neural networks (DNNs) in image classification, their vulnerability to adversarial attacks remains a critical challenge. Most existing detection methods rely on complex and poorly interpretable architectures, which compromise interpretability and generalization. To address this, we propose FeatureLens, a lightweight framework that acts as a lens to scrutinize anomalies in image features. Comprising an Image Feature Extractor (IFE) and shallow classifiers (e.g., SVM, MLP, or XGBoost) with model sizes ranging from 1,000 to 30,000 parameters, FeatureLens achieves high detection accuracy ranging from 97.8% to 99.75% in closed-set evaluation and 86.17% to 99.6% in generalization evaluation across FGSM, PGD, CW, and DAmageNet attacks, using only 51 dimensional features. By combining strong detection performance with excellent generalization, interpretability, and computational efficiency, FeatureLens offers a practical pathway toward transparent and effective adversarial defense.

摘要: 尽管深度神经网络（DNN）在图像分类方面表现出色，但其对对抗攻击的脆弱性仍然是一个严峻的挑战。大多数现有的检测方法依赖于复杂且难以解释的架构，这会损害可解释性和概括性。为了解决这个问题，我们提出了DeliverureLens，这是一个轻量级框架，可以充当镜头来检查图像特征中的异常。包括图像特征提取器（IFE）和浅层分类器（例如，ASM、MLP或XGboost）的模型大小范围为1，000至30，000个参数，EnterpriseLens在FGSM、PVD、CW和DAmageNet攻击中仅使用51维特征，在封闭集评估中实现了97.8%至99.75%的高检测准确率，在概括评估中实现了86.17%至99.6%的高检测准确率。通过将强大的检测性能与出色的概括性、可解释性和计算效率相结合，InspectureLens提供了一条实现透明有效对抗防御的实用途径。



## **14. Tuning for TraceTarnish: Techniques, Trends, and Testing Tangible Traits**

TraceTarnish调整：技术、趋势和测试纹理特征 cs.CR

20 pages, 8 figures, 2 tables

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03465v1) [paper-pdf](https://arxiv.org/pdf/2512.03465v1)

**Authors**: Robert Dilworth

**Abstract**: In this study, we more rigorously evaluated our attack script $\textit{TraceTarnish}$, which leverages adversarial stylometry principles to anonymize the authorship of text-based messages. To ensure the efficacy and utility of our attack, we sourced, processed, and analyzed Reddit comments--comments that were later alchemized into $\textit{TraceTarnish}$ data--to gain valuable insights. The transformed $\textit{TraceTarnish}$ data was then further augmented by $\textit{StyloMetrix}$ to manufacture stylometric features--features that were culled using the Information Gain criterion, leaving only the most informative, predictive, and discriminative ones. Our results found that function words and function word types ($L\_FUNC\_A$ $\&$ $L\_FUNC\_T$); content words and content word types ($L\_CONT\_A$ $\&$ $L\_CONT\_T$); and the Type-Token Ratio ($ST\_TYPE\_TOKEN\_RATIO\_LEMMAS$) yielded significant Information-Gain readings. The identified stylometric cues--function-word frequencies, content-word distributions, and the Type-Token Ratio--serve as reliable indicators of compromise (IoCs), revealing when a text has been deliberately altered to mask its true author. Similarly, these features could function as forensic beacons, alerting defenders to the presence of an adversarial stylometry attack; granted, in the absence of the original message, this signal may go largely unnoticed, as it appears to depend on a pre- and post-transformation comparison. "In trying to erase a trace, you often imprint a larger one." Armed with this understanding, we framed $\textit{TraceTarnish}$'s operations and outputs around these five isolated features, using them to conceptualize and implement enhancements that further strengthen the attack.

摘要: 在这项研究中，我们更严格地评估了我们的攻击脚本$\textit{TraceTarnish}$，该脚本利用对抗性样式学原则来匿名基于文本的消息的作者身份。为了确保攻击的有效性和实用性，我们对Reddit评论进行了来源、处理和分析，这些评论后来被子序列化为$\textit{TraceTarnish}$ data-以获得有价值的见解。然后，转换后的$\textit{TraceTarnish}$数据进一步由$\textit{StyloMeetup}$扩展，以制造文体特征--使用信息收益标准剔除的特征，只留下信息量最大、预测性最强和区分性最强的特征。我们的结果发现，功能词和功能词类型（$L\_FSYS\_A$\&$ $L\_FSYS\_T$）;内容词和内容词类型（$L\_CONT\_A$\&$L\_CONT\_T$）;和类型-令牌比（$ST\_GROUP\_TOKEN\_RATIO\_LEMMAS$）产生了显着的信息-收益读数。识别出的文体线索--功能词频率、内容词分布和类型-标记比--可以作为可靠的妥协指标（IoCs），揭示文本何时被故意更改以掩盖其真实作者。同样，这些功能可以充当法医信标，提醒防御者存在对抗性文体攻击;当然，在没有原始消息的情况下，这个信号可能在很大程度上被忽视，因为它似乎取决于转换前后的比较。“在试图擦除痕迹时，你通常会印上更大的痕迹。“有了这一理解，我们围绕这五个独立功能构建了$\textit{TraceTarnish}$的操作和输出，使用它们概念化和实施进一步加强攻击的增强功能。



## **15. Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems**

玩多米诺骨牌：对基于LLM的多代理系统的具有布局感知的多跳攻击 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04129v1) [paper-pdf](https://arxiv.org/pdf/2512.04129v1)

**Authors**: Ruichao Liang, Le Yin, Jing Chen, Cong Wu, Xiaoyu Zhang, Huangpeng Gu, Zijian Zhang, Yang Liu

**Abstract**: LLM-based multi-agent systems (MASs) have reshaped the digital landscape with their emergent coordination and problem-solving capabilities. However, current security evaluations of MASs are still confined to limited attack scenarios, leaving their security issues unclear and likely underestimated. To fill this gap, we propose TOMA, a topology-aware multi-hop attack scheme targeting MASs. By optimizing the propagation of contamination within the MAS topology and controlling the multi-hop diffusion of adversarial payloads originating from the environment, TOMA unveils new and effective attack vectors without requiring privileged access or direct agent manipulation. Experiments demonstrate attack success rates ranging from 40% to 78% across three state-of-the-art MAS architectures: \textsc{Magentic-One}, \textsc{LangManus}, and \textsc{OWL}, and five representative topologies, revealing intrinsic MAS vulnerabilities that may be overlooked by existing research. Inspired by these findings, we propose a conceptual defense framework based on topology trust, and prototype experiments show its effectiveness in blocking 94.8% of adaptive and composite attacks.

摘要: 基于LLM的多智能体系统（MAS）凭借其紧急协调和解决问题的能力重塑了数字格局。然而，当前对MAS的安全评估仍然仅限于有限的攻击场景，导致其安全问题不明确并且可能被低估。为了填补这一空白，我们提出了TOMA，这是一种针对MAS的、基于布局的多跳攻击方案。通过优化MAS布局内污染的传播并控制源自环境的对抗有效负载的多跳扩散，TOMA在不需要特权访问或直接代理操纵的情况下推出了新的有效攻击载体。实验表明，在三种最先进的MAS架构（\textsch {Magentic-One}、\textsch {LangManus}和\textsch {OWL}）以及五种代表性的布局中，攻击成功率从40%到78%不等，揭示了现有研究可能忽视的固有MAS漏洞。受这些发现的启发，我们提出了一个基于布局信任的概念防御框架，原型实验表明其在阻止94.8%的适应性和复合攻击方面的有效性。



## **16. Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models**

基于免疫记忆的越狱检测：大型语言模型的多代理自适应警卫 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03356v1) [paper-pdf](https://arxiv.org/pdf/2512.03356v1)

**Authors**: Jun Leng, Litian Zhang, Xi Zhang

**Abstract**: Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios.

摘要: 大型语言模型（LLM）已成为人工智能系统的基础，但它们仍然容易受到敌对越狱攻击。这些攻击涉及精心设计的提示，绕过安全护栏并诱导模型产生有害内容。因此，检测此类恶意输入查询对于维护LLM安全至关重要。现有的越狱检测方法通常涉及使用固定训练数据集将LLM微调为静态安全LLM。然而，这些方法在更新模型参数以提高鲁棒性时会产生巨大的计算成本，尤其是在面对新型越狱攻击时。受免疫记忆机制的启发，我们提出了用于越狱检测的多智能体自适应警卫（MAAG）框架。核心想法是为警卫配备记忆能力：在遇到新颖的越狱攻击时，系统会记住攻击模式，使其能够在未来遇到类似威胁时快速准确地识别出类似威胁。具体来说，MAAG首先从输入提示中提取激活值，并将其与存储在存储库中的历史激活进行比较，以进行快速初步检测。然后，防御代理根据这些检测结果模拟响应，辅助代理监督模拟过程，以提供检测结果的二次过滤。针对五个开源模型的广泛实验表明，MAAG的性能明显优于最先进的（SOTA）方法，在各种攻击场景中实现了98%的检测准确率和96%的F1评分。



## **17. Invasive Context Engineering to Control Large Language Models**

控制大型语言模型的侵入性上下文工程 cs.AI

4 pages

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03001v1) [paper-pdf](https://arxiv.org/pdf/2512.03001v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations.

摘要: 当前对大型语言模型操作员控制的研究通过对偏好示例、提示和输入/输出过滤进行训练，提高了模型针对对抗性攻击和不当行为的鲁棒性。尽管结果良好，但LLM仍然容易受到滥用，越狱可能性随着上下文长度的增加而增加。在长期背景下需要强有力的LLM安全保证。我们建议将控制句插入到LLM上下文中，作为侵入性上下文工程，以部分解决问题。我们建议这种技术可以推广到思想链过程中，以防止阴谋。侵入式上下文工程不依赖于LLM培训，从而避免了长期上下文情况的训练模型中出现的数据短缺陷阱。



## **18. Defense That Attacks: How Robust Models Become Better Attackers**

攻击的防御：稳健模型如何成为更好的攻击者 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02830v2) [paper-pdf](https://arxiv.org/pdf/2512.02830v2)

**Authors**: Mohamed Awad, Mahmoud Akrm, Walid Gomaa

**Abstract**: Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.

摘要: 深度学习在计算机视觉领域取得了巨大成功，但仍然容易受到对抗攻击。对抗训练是旨在提高模型稳健性的主要防御措施。然而，它对攻击可转移性的影响尚未得到充分研究。在这项工作中，我们询问对抗性训练是否无意中增加了对抗性示例的可移植性。为了解决这个问题，我们训练了一个由36个模型组成的多样化动物园，包括CNN和ViT，并进行了全面的可移植性实验。我们的结果揭示了一个明显的悖论：对抗训练（AT）模型会产生比标准模型更有效地传递的扰动，从而引入了新的生态系统风险。为了实现可重复性和进一步研究，我们发布了所有模型、代码和实验脚本。此外，我们认为稳健性评估不仅应该评估模型对转移攻击的抵抗力，还应该评估其产生可转移对抗示例的倾向。



## **19. Decryption Through Polynomial Ambiguity: Noise-Enhanced High-Memory Convolutional Codes for Post-Quantum Cryptography**

通过多项歧义解密：后量子密码学的降噪高内存卷积码 cs.CR

23 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2510.15515

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02822v2) [paper-pdf](https://arxiv.org/pdf/2512.02822v2)

**Authors**: Meir Ariel

**Abstract**: We present a novel approach to post-quantum cryptography that employs directed-graph decryption of noise-enhanced high-memory convolutional codes. The proposed construction generates random-like generator matrices that effectively conceal algebraic structure and resist known structural attacks. Security is further reinforced by the deliberate injection of strong noise during decryption, arising from polynomial division: while legitimate recipients retain polynomial-time decoding, adversaries face exponential-time complexity. As a result, the scheme achieves cryptanalytic security margins surpassing those of Classic McEliece by factors exceeding 2^(200). Beyond its enhanced security, the method offers greater design flexibility, supporting arbitrary plaintext lengths with linear-time decryption and uniform per-bit computational cost, enabling seamless scalability to long messages. Practical deployment is facilitated by parallel arrays of directed-graph decoders, which identify the correct plaintext through polynomial ambiguity while allowing efficient hardware and software implementations. Altogether, the scheme represents a compelling candidate for robust, scalable, and quantum-resistant public-key cryptography.

摘要: 我们提出了一种新的后量子密码学方法，该方法采用噪音增强的高内存卷积码的有向图解密。提出的结构生成类随机生成矩阵，可以有效地隐藏代数结构并抵抗已知的结构攻击。解密期间故意注入强噪音，这是由多项分解引起的，进一步加强了安全性：虽然合法接收者保留了多项时间解码，但对手面临着指数时间复杂性。结果，该计划实现的密码分析安全利润超过了Classic McEliece的安全利润，其系数超过了2^（200）。除了增强的安全性之外，该方法还提供了更大的设计灵活性，支持任意明文长度，具有线性时间解密和统一的每位计算成本，从而实现了对长消息的无缝可扩展性。有向图解码器的并行阵列促进了实际部署，这些解码器通过多元歧义识别正确的明文，同时允许高效的硬件和软件实现。总而言之，该方案代表了稳健、可扩展和抗量子公钥加密的令人信服的候选者。



## **20. Characterizing Cyber Attacks against Space Infrastructures with Missing Data: Framework and Case Study**

描述针对缺失数据的太空基础设施的网络攻击：框架和案例研究 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02414v1) [paper-pdf](https://arxiv.org/pdf/2512.02414v1)

**Authors**: Ekzhin Ear, Jose Luis Castanon Remy, Caleb Chang, Qiren Que, Antonia Feffer, Shouhuai Xu

**Abstract**: Cybersecurity of space infrastructures is an emerging topic, despite space-related cybersecurity incidents occurring as early as 1977 (i.e., hijacking of a satellite transmission signal). There is no single dataset that documents cyber attacks against space infrastructures that have occurred in the past; instead, these incidents are often scattered in media reports while missing many details, which we dub the missing-data problem. Nevertheless, even ``low-quality'' datasets containing such reports would be extremely valuable because of the dearth of space cybersecurity data and the sensitivity of space infrastructures which are often restricted from disclosure by governments. This prompts a research question: How can we characterize real-world cyber attacks against space infrastructures? In this paper, we address the problem by proposing a framework, including metrics, while also addressing the missing-data problem by leveraging methodologies such as the Space Attack Research and Tactic Analysis (SPARTA) and the Adversarial Tactics, Techniques, and Common Knowledge (ATT&CK) to ``extrapolate'' the missing data in a principled fashion. We show how the extrapolated data can be used to reconstruct ``hypothetical but plausible'' space cyber kill chains and space cyber attack campaigns that have occurred in practice. To show the usefulness of the framework, we extract data for 108 cyber attacks against space infrastructures and show how to extrapolate this ``low-quality'' dataset containing missing information to derive 6,206 attack technique-level space cyber kill chains. Our findings include: cyber attacks against space infrastructures are getting increasingly sophisticated; successful protection of the link segment between the space and user segments could have thwarted nearly half of the 108 attacks. We will make our dataset available.

摘要: 尽管早在1977年就发生过与太空相关的网络安全事件（即，劫持卫星传输信号）。没有一个数据集可以记录过去发生的针对太空基础设施的网络攻击;相反，这些事件通常分散在媒体报道中，同时缺少许多细节，我们称之为数据丢失问题。然而，即使是包含此类报告的“低质量”数据集也极具价值，因为太空网络安全数据缺乏，而且太空基础设施的敏感性往往受到政府的限制。这引发了一个研究问题：我们如何描述现实世界中针对太空基础设施的网络攻击？在本文中，我们通过提出一个包括指标在内的框架来解决这个问题，同时还通过利用空间攻击研究和战术分析（SPARTA）和对抗战术、技术和常识（ATT & CK）等方法来解决丢失数据的问题，以有原则的方式“推断”丢失的数据。我们展示了如何使用推断的数据来重建“假设但可能”的太空网络杀伤链和实践中发生的太空网络攻击活动。为了展示该框架的有用性，我们提取了针对太空基础设施的108次网络攻击的数据，并展示了如何推断这个包含缺失信息的“低质量”数据集，以推导出6，206个攻击技术级别的太空网络杀伤链。我们的调查结果包括：针对太空基础设施的网络攻击正变得越来越复杂;成功保护太空和用户段之间的链路段可以挫败108次攻击中的近一半。我们将提供我们的数据集。



## **21. LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems**

LeechHijack：智能代理系统中的秘密计算资源开发 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02321v1) [paper-pdf](https://arxiv.org/pdf/2512.02321v1)

**Authors**: Yuanhe Zhang, Weiliu Wang, Zhenhong Zhou, Kun Wang, Jie Zhang, Li Sun, Yang Liu, Sen Su

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.

摘要: 基于大型语言模型（LLM）的代理在推理、规划和工具使用方面表现出了非凡的能力。最近提出的模型上下文协议（HCP）已成为将外部工具集成到代理系统中的统一框架，从而实现社区构建功能的蓬勃发展的开放生态系统。然而，使HCP具有吸引力的开放性和可组合性也引入了一个关键但被忽视的安全假设--对第三方工具提供商的隐性信任。在这项工作中，我们识别并正式化一类新的攻击，这些攻击利用此信任边界，而不违反显式许可。我们将这种新的攻击向量称为隐式毒性，其中恶意行为完全发生在允许的权限范围内。我们提出了LeechHijack，一个潜在的嵌入式利用计算劫持，其中一个敌对的MCP工具隐蔽地征用代理的计算资源用于未经授权的工作负载。LeechHijack通过两个阶段的机制运行：植入阶段，在工具中嵌入一个看起来很好的后门，以及利用阶段，后门在预定义的触发器上激活以建立命令和控制通道。通过此通道，攻击者注入代理执行的额外任务，就好像它们是其正常工作流的一部分，有效地寄生了用户的计算预算。我们在四个主要LLM家族中实施LeechHijack。实验表明，LeechHijack的平均成功率为77.25%，与基线相比资源消耗为18.62%。这项研究强调了迫切需要计算出处和资源证明机制来保护新兴的LCP生态系统。



## **22. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers**

认知：从评估到防御多模式LLM验证码解决器 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02318v2) [paper-pdf](https://arxiv.org/pdf/2512.02318v2)

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation pipeline.Code Availability (https://anonymous.4open.science/r/Captcha-465E/).

摘要: 本文研究了多模式大型语言模型（MLLM）如何破坏视觉验证码的安全保证。我们确定了对手可以使用现成模型廉价地自动化验证码解决的攻击表面。我们评估了18种现实世界的CAPTCHA任务类型中的7种领先的商业和开源MLLM，衡量单次准确性、有限再试下的成功率、端到端延迟和每次解决的成本。我们进一步分析了特定任务的即时工程和少数镜头演示对求解器有效性的影响。我们发现，MLLM可以以类似于人类的成本和延迟可靠地解决面向认知和低交互性的CAPTCHA任务，而对于当前的模型来说，需要细粒度本地化、多步空间推理或跨框架一致性的任务仍然明显困难。通过检查此类MLLM的推理痕迹，我们研究模型为何在特定验证码难题上成功/失败的潜在机制，并利用这些见解来得出选择和加强验证码任务的防御导向指南。最后，我们讨论了部署CAPTCHA作为其虐待缓解管道的一部分对平台运营商的影响。代码可用性（https：//anonymous.4open.science/r/Captcha-465E/）。



## **23. Adversarial Robustness of Traffic Classification under Resource Constraints: Input Structure Matters**

资源约束下流量分类的对抗鲁棒性：输入结构很重要 cs.NI

Accepted at the 2025 IEEE International Symposium on Networks, Computers and Communications (ISNCC)

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.02276v1) [paper-pdf](https://arxiv.org/pdf/2512.02276v1)

**Authors**: Adel Chehade, Edoardo Ragusa, Paolo Gastaldo, Rodolfo Zunino

**Abstract**: Traffic classification (TC) plays a critical role in cybersecurity, particularly in IoT and embedded contexts, where inspection must often occur locally under tight hardware constraints. We use hardware-aware neural architecture search (HW-NAS) to derive lightweight TC models that are accurate, efficient, and deployable on edge platforms. Two input formats are considered: a flattened byte sequence and a 2D packet-wise time series; we examine how input structure affects adversarial vulnerability when using resource-constrained models. Robustness is assessed against white-box attacks, specifically Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). On USTC-TFC2016, both HW-NAS models achieve over 99% clean-data accuracy while remaining within 65k parameters and 2M FLOPs. Yet under perturbations of strength 0.1, their robustness diverges: the flat model retains over 85% accuracy, while the time-series variant drops below 35%. Adversarial fine-tuning delivers robust gains, with flat-input accuracy exceeding 96% and the time-series variant recovering over 60 percentage points in robustness, all without compromising efficiency. The results underscore how input structure influences adversarial vulnerability, and show that even compact, resource-efficient models can attain strong robustness, supporting their practical deployment in secure edge-based TC.

摘要: 流量分类（TC）在网络安全中发挥着关键作用，特别是在物联网和嵌入式环境中，检查通常必须在严格的硬件限制下在本地进行。我们使用硬件感知神经架构搜索（HW-NAS）来推导准确、高效且可在边缘平台上部署的轻量级TC模型。考虑了两种输入格式：扁平字节序列和2D逐包时间序列;我们研究了使用资源受限模型时输入结构如何影响对抗脆弱性。鲁棒性是针对白盒攻击进行评估的，特别是快速梯度符号法（FGSM）和投影梯度下降（PVD）。在USTC-TFC 2016上，这两种HW-NAS型号都实现了超过99%的干净数据准确性，同时保持在65 k个参数和2 M个FLOP范围内。然而，在强度0.1的扰动下，它们的鲁棒性出现了分歧：平坦模型保留了85%以上的准确性，而时间序列变体则降至35%以下。对抗性微调可带来稳健的收益，平坦输入准确性超过96%，时间序列变量的稳健性恢复超过60个百分点，而所有这些都不会影响效率。结果强调了输入结构如何影响对抗脆弱性，并表明即使是紧凑、资源高效的模型也可以获得强大的鲁棒性，支持其在安全的基于边缘的TC中的实际部署。



## **24. TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?**

TradeTrap：基于LLM的贸易代理真的可靠和忠诚吗？ cs.AI

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.02261v1) [paper-pdf](https://arxiv.org/pdf/2512.02261v1)

**Authors**: Lewen Yan, Jilin Mei, Tianyi Zhou, Lige Huang, Jie Zhang, Dongrui Liu, Jing Shao

**Abstract**: LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at https://github.com/Yanlewen/TradeTrap.

摘要: 基于LLM的交易代理越来越多地部署在现实世界的金融市场中，以执行自主分析和执行。然而，尽管它们在高风险、不可逆转的金融环境中运营，但它们在对抗或故障条件下的可靠性和稳健性在很大程度上仍未得到检验。我们提出TradeTrap，这是一个统一的评估框架，用于系统性地对适应性和程序性自主交易代理进行压力测试。TradeTrap针对自主交易代理的四个核心组件：市场情报、策略制定、投资组合和分类帐处理以及交易执行，并评估其在受控系统级扰动下的稳健性。所有评估都是在闭环历史回溯测试环境中对真实美国股市数据进行的，初始条件相同，从而实现对代理人和攻击进行公平且可重复的比较。大量实验表明，单一成分的微小扰动可以通过代理人决策循环传播，并在两种代理人类型中引发极端集中、失控的风险敞口和大规模投资组合提款，这表明当前的自主交易代理人可能会在系统层面被系统性地误导。我们的代码可以在https://github.com/Yanlewen/TradeTrap上找到。



## **25. Physical ID-Transfer Attacks against Multi-Object Tracking via Adversarial Trajectory**

通过对抗轨迹对多目标跟踪的物理ID传输攻击 cs.CV

Accepted to Annual Computer Security Applications Conference (ACSAC) 2024

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01934v1) [paper-pdf](https://arxiv.org/pdf/2512.01934v1)

**Authors**: Chenyi Wang, Yanmao Man, Raymond Muller, Ming Li, Z. Berkay Celik, Ryan Gerdes, Jonathan Petit

**Abstract**: Multi-Object Tracking (MOT) is a critical task in computer vision, with applications ranging from surveillance systems to autonomous driving. However, threats to MOT algorithms have yet been widely studied. In particular, incorrect association between the tracked objects and their assigned IDs can lead to severe consequences, such as wrong trajectory predictions. Previous attacks against MOT either focused on hijacking the trackers of individual objects, or manipulating the tracker IDs in MOT by attacking the integrated object detection (OD) module in the digital domain, which are model-specific, non-robust, and only able to affect specific samples in offline datasets. In this paper, we present AdvTraj, the first online and physical ID-manipulation attack against tracking-by-detection MOT, in which an attacker uses adversarial trajectories to transfer its ID to a targeted object to confuse the tracking system, without attacking OD. Our simulation results in CARLA show that AdvTraj can fool ID assignments with 100% success rate in various scenarios for white-box attacks against SORT, which also have high attack transferability (up to 93% attack success rate) against state-of-the-art (SOTA) MOT algorithms due to their common design principles. We characterize the patterns of trajectories generated by AdvTraj and propose two universal adversarial maneuvers that can be performed by a human walker/driver in daily scenarios. Our work reveals under-explored weaknesses in the object association phase of SOTA MOT systems, and provides insights into enhancing the robustness of such systems.

摘要: 多目标跟踪（MOT）是计算机视觉中的一项关键任务，其应用范围从监控系统到自动驾驶。然而，MOT算法面临的威胁尚未得到广泛研究。特别是，被跟踪对象与其分配的ID之间的不正确关联可能会导致严重的后果，例如错误的轨迹预测。之前针对MOT的攻击要么集中在劫持单个对象的跟踪器上，要么通过攻击数字域中的集成对象检测（OD）模块来操纵MOT中的跟踪器ID，这些模块是特定于模型的、不鲁棒的，并且只能影响离线数据集中的特定样本。在本文中，我们介绍了AdvTraj，这是第一个针对检测跟踪MOT的在线和物理ID操纵攻击，其中攻击者使用对抗轨迹将其ID传输到目标对象以混淆跟踪系统，而不攻击OD。我们在CARLA中的模拟结果表明，AdvTraj可以在针对SORT的白盒攻击的各种场景中以100%的成功率欺骗ID分配，由于其共同的设计原则，SOTA还具有针对最先进（SOTA）MOT算法的高攻击转移性（高达93%的攻击成功率）。我们描述了AdvTraj生成的轨迹模式，并提出了两种可由人类步行者/驾驶员在日常场景中执行的通用对抗动作。我们的工作揭示了SOTA MOT系统对象关联阶段未充分探索的弱点，并提供了增强此类系统鲁棒性的见解。



## **26. Many-to-One Adversarial Consensus: Exposing Multi-Agent Collusion Risks in AI-Based Healthcare**

多对一对抗性共识：暴露基于人工智能的医疗保健中的多代理合谋风险 cs.CR

7 pages Conference level paper

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03097v1) [paper-pdf](https://arxiv.org/pdf/2512.03097v1)

**Authors**: Adeela Bashir, The Anh han, Zia Ush Shamszaman

**Abstract**: The integration of large language models (LLMs) into healthcare IoT systems promises faster decisions and improved medical support. LLMs are also deployed as multi-agent teams to assist AI doctors by debating, voting, or advising on decisions. However, when multiple assistant agents interact, coordinated adversaries can collude to create false consensus, pushing an AI doctor toward harmful prescriptions. We develop an experimental framework with scripted and unscripted doctor agents, adversarial assistants, and a verifier agent that checks decisions against clinical guidelines. Using 50 representative clinical questions, we find that collusion drives the Attack Success Rate (ASR) and Harmful Recommendation Rates (HRR) up to 100% in unprotected systems. In contrast, the verifier agent restores 100% accuracy by blocking adversarial consensus. This work provides the first systematic evidence of collusion risk in AI healthcare and demonstrates a practical, lightweight defence that ensures guideline fidelity.

摘要: 将大型语言模型（LLM）集成到医疗保健物联网系统中有望实现更快的决策并改善医疗支持。LLM还被部署为多代理团队，通过辩论、投票或就决策提供建议来协助人工智能医生。然而，当多个助理特工互动时，协调一致的对手可能会勾结以建立错误共识，将人工智能医生推向有害的处方。我们开发了一个实验框架，其中包含有脚本和无脚本的医生代理、对抗助理和根据临床指南检查决策的验证者代理。通过使用50个代表性的临床问题，我们发现共谋导致未受保护的系统中的攻击成功率（ASB）和有害推荐率（HRR）高达100%。相比之下，验证者代理通过阻止对抗共识来恢复100%的准确性。这项工作提供了人工智能医疗保健中共谋风险的第一个系统性证据，并展示了一种实用、轻量级的防御，可以确保指南的忠实性。



## **27. Securing Large Language Models (LLMs) from Prompt Injection Attacks**

保护大型语言模型（LLM）免受提示注入攻击 cs.CR

10 pages, 1 figure, 1 table

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01326v1) [paper-pdf](https://arxiv.org/pdf/2512.01326v1)

**Authors**: Omar Farooq Khan Suri, John McCrae

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.

摘要: 大型语言模型（LLM）越来越多地被部署在现实世界的应用程序中，但它们的灵活性使它们容易受到提示注入攻击。这些攻击利用模型的描述跟踪能力使其执行恶意任务。最近的工作提出了JATMO，这是一种针对任务的微调方法，它训练非指令调优的基本模型来执行单一功能，从而减少对对抗性指令的敏感性。在这项研究中，我们评估了JATMO对HOUYI的稳健性，HOUYI是一种系统性突变和优化对抗提示的基因攻击框架。我们通过引入自定义适应度评分、修改后的突变逻辑和用于本地模型测试的新工具来调整HOUYI，从而能够更准确地评估防御有效性。我们根据JATMO方法对LLaMA 2- 7 B、Qwen 1.5 - 4 B和Qwen 1.5 -0.5B模型进行了微调，并将它们与微调的GPT-3.5-Turbo基线进行了比较。结果表明，虽然JATMO相对于经描述调整的模型降低了攻击成功率，但它并不能完全阻止注入;利用多语言线索或代码相关破坏者的对手仍然绕过防御。我们还观察到生成质量和注入脆弱性之间的权衡，表明更好的任务性能通常与易感性的增加相关。我们的结果强调了基于微调的防御的前景和局限性，并指出需要分层、了解对手情况的缓解策略。



## **28. DPAC: Distribution-Preserving Adversarial Control for Diffusion Sampling**

DPAC：扩散抽样的分布保持对抗控制 cs.CV

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01153v1) [paper-pdf](https://arxiv.org/pdf/2512.01153v1)

**Authors**: Han-Jin Lee, Han-Ju Lee, Jin-Seong Kim, Seok-Hwan Choi

**Abstract**: Adversarially guided diffusion sampling often achieves the target class, but sample quality degrades as deviations between the adversarially controlled and nominal trajectories accumulate. We formalize this degradation as a path-space Kullback-Leibler divergence(path-KL) between controlled and nominal (uncontrolled) diffusion processes, thereby showing via Girsanov's theorem that it exactly equals the control energy. Building on this stochastic optimal control (SOC) view, we theoretically establish that minimizing this path-KL simultaneously tightens upper bounds on both the 2-Wasserstein distance and Fréchet Inception Distance (FID), revealing a principled connection between adversarial control energy and perceptual fidelity. From a variational perspective, we derive a first-order optimality condition for the control: among all directions that yield the same classification gain, the component tangent to iso-(log-)density surfaces (i.e., orthogonal to the score) minimizes path-KL, whereas the normal component directly increases distributional drift. This leads to DPAC (Distribution-Preserving Adversarial Control), a diffusion guidance rule that projects adversarial gradients onto the tangent space defined by the generative score geometry. We further show that in discrete solvers, the tangent projection cancels the O(Δt) leading error term in the Wasserstein distance, achieving an O(Δt^2) quality gap; moreover, it remains second-order robust to score or metric approximation. Empirical studies on ImageNet-100 validate the theoretical predictions, confirming that DPAC achieves lower FID and estimated path-KL at matched attack success rates.

摘要: 不利引导的扩散采样通常可以达到目标类别，但随着不利控制轨迹和名义轨迹之间偏差的积累，样本质量会下降。我们将这种退化形式化为受控和名义（非受控）扩散过程之间的路径空间Kullback-Leibler分歧（路径-KL），从而通过吉萨诺夫定理表明它完全等于控制能。基于这种随机最优控制（SOC）观点，我们从理论上确定，最小化这条路径KL同时收紧了2-Wasserstein距离和Fréchet初始距离（DID）的上界，揭示了对抗控制能量和感知保真度之间的原则联系。从变分的角度来看，我们推导出控制的一阶最优性条件：在产生相同分类收益的所有方向中，与等（log）密度表面切向的分量（即，与分数垂直）最小化路径KL，而正向分量直接增加分布漂移。这导致了DPAC（分布保持对抗控制），这是一种扩散指导规则，将对抗梯度投影到生成分数几何定义的切空间上。我们进一步表明，在离散求解器中，切向投影消除了沃瑟斯坦距离中的O（Δt）领先误差项，从而实现了O（Δ t ' 2）质量差距;此外，它对得分或度量逼近保持二阶鲁棒性。ImageNet-100上的实证研究验证了理论预测，证实DPAC在匹配的攻击成功率下实现了较低的DID和估计路径KL。



## **29. The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches**

欺骗概述：使用边缘补丁对交通标志进行物理对抗攻击 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.00765v2) [paper-pdf](https://arxiv.org/pdf/2512.00765v2)

**Authors**: Haojie Ji, Te Hu, Haowen Li, Long Jin, Chongshi Xin, Yuchi Yao, Jiarui Xiao

**Abstract**: Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.

摘要: 智能驾驶系统很容易受到交通标志的物理对抗攻击。这些攻击可能会导致错误分类，导致错误的驾驶决定，从而危及道路安全。此外，在V2X网络中，此类误解可能会传播，引发连锁故障，从而扰乱整体流量和系统稳定性。然而，当前物理攻击的一个关键局限性是缺乏隐形性。大多数方法都会对标志的中心区域施加扰动，从而产生人类观察者很容易检测到的视觉上明显的模式，从而限制了它们在现实世界中的实用性。这项研究提出了TESP-Attack，这是一种用于交通标志分类的新型隐形感知对抗补丁方法。根据人类视觉注意力主要集中在交通标志的中心区域的观察，我们采用实例分割来生成符合标志形状特征的边缘对齐的面罩。使用U-Net生成器来制作对抗补丁，然后通过颜色和纹理约束以及频域分析对其进行优化，以实现与背景环境的无缝集成，从而实现高效的视觉隐藏。所提出的方法在具有不同架构的交通标志分类模型中表现出出色的攻击成功率，在有限的查询预算下实现了90%以上。它还展现出强大的跨模型可移植性，并保持稳健的现实世界性能，在不同角度和距离下保持稳定。



## **30. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

对抗性混乱攻击：扰乱多模式大型语言模型 cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.20494v3) [paper-pdf](https://arxiv.org/pdf/2511.20494v3)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Practical applications include embedding such adversarial images into websites to prevent MLLM-powered AI Agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and Adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.

摘要: 我们引入了对抗性混乱攻击，这是针对多模式大型语言模型（MLLM）的一类新型威胁。与越狱或有针对性的错误分类不同，目标是引发系统性破坏，使模型生成不连贯或自信地错误的输出。实际应用包括将这种对抗性图像嵌入网站，以防止MLLM驱动的AI代理可靠地运行。拟议的攻击使用一小部分开源MLLM来最大化下一个令牌的熵。在白盒设置中，我们表明，单个对抗图像可以扰乱集合中的所有模型，无论是在完整图像还是对抗验证码设置中。尽管依赖于基本的对抗技术（PVD），但攻击会产生转移到两个看不见的开源的扰动（例如，Qwen 3-DL）和专有（例如，GPT-5.1）型号。



## **31. Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives**

泄露豆子：教法学硕士自我报告他们隐藏的目标 cs.AI

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2511.06626v4) [paper-pdf](https://arxiv.org/pdf/2511.06626v4)

**Authors**: Chloe Li, Mary Phuong, Daniel Tan

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to occasionally make factual mistakes, then admit them when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AIs.

摘要: 随着人工智能系统变得更有能力执行复杂的代理任务，它们也变得更有能力追求不想要的目标并造成伤害。之前的工作试图通过直接询问模型的目标和行为来捕捉这些不安全的实例。然而，信任审讯的主要弱点是模型可能会撒谎。我们提出了自我报告微调（SRFT），这是一种简单的监督微调技术，可以训练模型偶尔犯事实错误，然后在被问及时承认错误。我们表明，在简单的问答环境中承认事实错误将分配失调（OOD）推广为在对抗性代理环境中承认隐藏的错位目标。我们在OOD隐形任务中评估SRFT，其中模型被指示完成隐藏的未对齐目标以及用户指定的目标，而不会被监控发现。在SRFT之后，模特们在被审问时更有可能坦白其隐藏目标的细节，即使在不披露这些目标的强大压力下也是如此。SRFT模型上的询问可以检测出具有接近天花板性能的隐藏目标（F1评分= 0.98），而基线模型在相同条件下询问时则存在缺陷（F1评分= 0）。SRFT模型上的询问可以进一步获取隐藏目标的内容，恢复28-100%的细节，而在基线模型和预填充助理回合攻击中恢复的细节为0%。这提供了一种有希望的技术，可以促进诚实倾向并将不一致的人工智能定罪。



## **32. Reasoning Up the Instruction Ladder for Controllable Language Models**

可控语言模型的指令阶梯推理 cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.04694v3) [paper-pdf](https://arxiv.org/pdf/2511.04694v3)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ~7K aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.

摘要: 随着基于大型语言模型（LLM）的系统在现实世界的决策中扮演着高风险的角色，它们必须协调来自多个来源的竞争指令（例如，模型开发人员、用户和工具）在单个提示上下文中。因此，在LLM中强制执行指令层次结构（IHS）（其中更高级的指令优先于较低优先级的请求）对于LLM的可靠性和可控性至关重要。在这项工作中，我们将指令层次结构分解重新构建为一项推理任务。具体来说，模型必须在生成响应之前首先“思考”给定用户提示和更高优先级（系统）指令之间的关系。为了通过训练实现这种能力，我们构建了VerIHS，这是一个具有可验证答案的约束遵循任务的指令层次数据集。此数据集包括约7K个对齐且冲突的系统用户指令。我们表明，使用VerIHS的轻量级强化学习可以有效地将模型的一般推理能力转移到指令优先级。我们的微调模型在指令遵循和指令层次基准方面实现了一致的改进，在IHEval冲突设置方面实现了大约20%的改进。这种推理能力还推广到培训分布以外的安全关键环境。通过将安全问题视为解决敌对用户输入和预定义的高优先级策略之间的冲突，我们训练的模型增强了针对越狱和即时注入攻击的鲁棒性，将攻击成功率（ASB）降低高达20%。这些结果表明，对指令层次结构的推理提供了一条通往可靠LLM的实用途径，其中对系统提示的更新会产生模型行为的可控且稳健的变化。



## **33. Bilevel Models for Adversarial Learning and A Case Study**

对抗学习的两层模型及案例研究 cs.LG

This paper has been accepted by Mathematics

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2510.25121v2) [paper-pdf](https://arxiv.org/pdf/2510.25121v2)

**Authors**: Yutong Zheng, Qingna Li

**Abstract**: Adversarial learning has been attracting more and more attention thanks to the fast development of machine learning and artificial intelligence. However, due to the complicated structure of most machine learning models, the mechanism of adversarial attacks is not well interpreted. How to measure the effect of attacks is still not quite clear.   In this paper, we investigate the adversarial learning from the perturbation analysis point of view.   We characterize the robustness of learning models through the calmness of the solution mapping.   In the case of convex clustering models, we identify the conditions under which the clustering results remain the same under perturbations.   When the noise level is large, it leads to an attack.   Therefore, we propose two bilevel models for adversarial learning where the effect of adversarial learning is measured   by some deviation function.   Specifically, we systematically study the so-called $δ$-measure and show that under certain conditions, it can be used as a deviation function in adversarial learning for convex clustering models.   Finally, we conduct numerical tests to verify the above theoretical results as well as the efficiency of the two proposed bilevel models.

摘要: 随着机器学习和人工智能的快速发展，对抗学习受到越来越多的关注。然而，由于大多数机器学习模型的结构复杂，对抗性攻击的机制没有得到很好的解释。如何衡量攻击的效果仍然不太清楚。   本文从扰动分析的角度研究了对抗学习。   我们通过解决方案映射的平静来描述学习模型的稳健性。   在凸集群模型的情况下，我们确定了集群结果在扰动下保持相同的条件。   当噪音水平很大时，会导致攻击。   因此，我们提出了两个对抗学习的双层模型，其中衡量对抗学习的效果   通过某种偏差函数。   具体来说，我们系统地研究了所谓的$δ$-测量，并表明在一定条件下，它可以用作凸集群模型对抗学习中的偏差函数。   最后，我们进行了数值测试来验证上述理论结果以及所提出的两个双层模型的有效性。



## **34. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

获得丰富或模具缩放：有利可图的交易推理计算的鲁棒性 cs.LG

21 pages

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2510.06790v2) [paper-pdf](https://arxiv.org/pdf/2510.06790v2)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization. For example, InternVL 3.5 gpt-oss 20B gains little robustness when its test compute is scaled, but such scaling adds significant robustness if we first robustify its vision encoder. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Thus, we advise layering train-time and test-time defenses to obtain their synergistic benefit.

摘要: 尽管模型的鲁棒性投入了大量的训练计算投资，但它们仍然容易受到不利的分布外（OOD）数据的影响。Zaremba等人（2025）在测试时在这个问题上取得了进展，表明LLM推理提高了旨在阻止攻击的模型规范的满意度，从而导致推理工作量和越狱稳健性之间的相关性。然而，当攻击者能够访问梯度或多模式输入时，测试计算的这种好处就会消失。我们解决了这一差距，澄清了即使在这种情况下，推理计算也能带来好处。我们的方法认为，组合概括（OOD数据可以通过其内分布（ID）组件来理解）使得能够遵守针对敌对OOD输入的防御规范。也就是说，我们从推理计算假设（RICH）中验证了鲁棒性：由于模型的训练数据更好地反映了受攻击数据的成分，推理计算防御会获利。我们在视觉语言模型和攻击类型中从经验上支持了这一假设，如果OOD数据上的规范通过组合概括解锁，则可以从测试时计算中找到鲁棒性的收益。例如，InternVL3.5 gtt-oss 20 B在扩展其测试计算时几乎没有获得鲁棒性，但如果我们首先对其视觉编码器进行鲁棒性验证，这种扩展会增加显着的鲁棒性。推理计算的稳健性优势与基础模型稳健性的这种相关性是RICH的富而富的动态：受攻击的数据组件对于稳健模型来说更具ID，有助于组合概括OOD数据。因此，我们建议将训练时和测试时防御分层，以获得协同效益。



## **35. ZQBA: Zero Query Black-box Adversarial Attack**

ZQBA：零查询黑匣子对抗攻击 cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.

摘要: 当前的黑匣子对抗攻击要么需要多个查询，要么需要扩散模型来产生可能损害目标模型性能的对抗样本。然而，这些方法需要训练替代损失或扩散模型来产生对抗样本，这限制了它们在现实世界环境中的适用性。因此，我们提出了零查询黑匣子对抗（ZQBA）攻击，利用深度神经网络（DNN）的表示来愚弄其他网络。我们不需要数千个查询来产生具有欺骗性的对抗样本，而是使用从DNN获得的特征地图并将它们添加到干净的图像中，以损害目标模型的分类。结果表明，ZQBA可以将对抗样本传输到不同的模型和各种数据集，即CIFAR和Tiny ImageNet。实验还表明，ZQBA比单个查询的最先进的黑匣子攻击更有效，同时保持了扰动的不可感知性，并进行了定量（SSIM）和定性评估，强调了在现实世界中使用DNN的漏洞。所有源代码均可在https://github.com/Joana-Cabral/ZQBA上获取。



## **36. Less is More: Towards Simple Graph Contrastive Learning**

少即是多：走向简单图形对比学习 cs.LG

Submitted to ICLR 2026

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2509.25742v2) [paper-pdf](https://arxiv.org/pdf/2509.25742v2)

**Authors**: Yanan Zhao, Feng Ji, Jingyang Dai, Jiaze Ma, Wee Peng Tay

**Abstract**: Graph Contrastive Learning (GCL) has shown strong promise for unsupervised graph representation learning, yet its effectiveness on heterophilic graphs, where connected nodes often belong to different classes, remains limited. Most existing methods rely on complex augmentation schemes, intricate encoders, or negative sampling, which raises the question of whether such complexity is truly necessary in this challenging setting. In this work, we revisit the foundations of supervised and unsupervised learning on graphs and uncover a simple yet effective principle for GCL: mitigating node feature noise by aggregating it with structural features derived from the graph topology. This observation suggests that the original node features and the graph structure naturally provide two complementary views for contrastive learning. Building on this insight, we propose an embarrassingly simple GCL model that uses a GCN encoder to capture structural features and an MLP encoder to isolate node feature noise. Our design requires neither data augmentation nor negative sampling, yet achieves state-of-the-art results on heterophilic benchmarks with minimal computational and memory overhead, while also offering advantages in homophilic graphs in terms of complexity, scalability, and robustness. We provide theoretical justification for our approach and validate its effectiveness through extensive experiments, including robustness evaluations against both black-box and white-box adversarial attacks.

摘要: 图对比学习（GCL）在无监督图表示学习方面表现出了强大的前景，但它对异性图（其中连接的节点通常属于不同类别）的有效性仍然有限。大多数现有的方法依赖于复杂的增强方案、复杂的编码器或负采样，这引发了这样的复杂性在这种具有挑战性的环境中是否确实必要的问题。在这项工作中，我们重新审视了图上有监督和无监督学习的基础，并揭示了GCL的一个简单而有效的原则：通过将节点特征噪音与从图布局中衍生的结构特征聚合来减轻节点特征噪音。这一观察表明，原始的节点特征和图结构自然地为对比学习提供了两个补充的视图。基于这一见解，我们提出了一个极其简单的GCL模型，该模型使用GCN编码器来捕获结构特征，并使用MLP编码器来隔离节点特征噪音。我们的设计既不需要数据增强，也不需要负采样，但以最少的计算和内存负担在异嗜基准上实现了最先进的结果，同时还在同嗜图中提供了复杂性、可扩展性和鲁棒性方面的优势。我们为我们的方法提供了理论依据，并通过大量实验验证了其有效性，包括针对黑匣子和白盒对抗攻击的鲁棒性评估。



## **37. Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail**

通过峰值神经网络梯度稀疏度追踪实现准确性与稳健性权衡 cs.NE

Work under peer-review

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2509.23762v3) [paper-pdf](https://arxiv.org/pdf/2509.23762v3)

**Authors**: Luu Trong Nhan, Luu Trung Duong, Pham Ngoc Nam, Truong Cong Thang

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, (particularly for vision-related tasks) remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks. Our findings offer new insights into the dual role of gradient sparsity in SNN training.

摘要: 尖峰神经网络（SNN）引起了计算神经科学和人工智能日益增长的兴趣，主要是由于其固有的能源效率和紧凑的内存占用。然而，在SNN中实现对抗稳健性（特别是对于视觉相关任务）仍然是一个新生且未充分探索的挑战。最近的研究提出利用稀疏梯度作为一种正规化形式，以增强针对对抗性扰动的鲁棒性。在这项工作中，我们提出了一个令人惊讶的发现：在特定的架构配置下，SNN表现出自然的梯度稀疏性，并且可以在不需要任何显式正规化的情况下实现最先进的对抗防御性能。进一步的分析揭示了鲁棒性和概括性之间的权衡：虽然稀疏梯度有助于提高对抗弹性，但它们可能会损害模型的概括能力;相反，更密集的梯度支持更好的概括性，但会增加对攻击的脆弱性。我们的研究结果为梯度稀疏性在SNN训练中的双重作用提供了新的见解。



## **38. Observation-Free Attacks on Online Learning to Rank**

对在线学习排名的无观察攻击 cs.LG

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2509.22855v4) [paper-pdf](https://arxiv.org/pdf/2509.22855v4)

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Moharir

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data.

摘要: 在线排名学习（OLTR）在信息检索和机器学习系统中发挥着至关重要的作用，在搜索引擎和内容排序器中有广泛的应用。然而，尽管OLTR算法被广泛采用，但人们对OLTR算法对协同对抗攻击的敏感性仍然知之甚少。在这项工作中，我们提出了一个新颖的框架来攻击一些广泛使用的OLTR算法。我们的框架旨在推广一组目标项，使它们出现在T-o（T）轮的前K推荐列表中，同时在学习算法中引发线性遗憾。我们提出了两种新颖的攻击策略：针对CascadeUCB 1的CascadeOFA和针对PBM-UCB的PBMOFA。我们提供了理论保证，表明这两种策略只需要O（log T）操作即可成功。此外，我们还通过现实世界数据的实证结果来补充理论分析。



## **39. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

当广告成为简介：利用LLM揭露大规模网络广告的隐形风险 cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.

摘要: 对显式定位的监管限制并没有消除网络上的算法分析，因为优化系统仍然根据用户的私人属性调整广告交付。强大的零镜头多模式大型语言模型（LLM）的广泛使用极大地降低了利用这些潜在信号进行对抗性推理的障碍。我们调查这种新出现的社会风险，特别是对手现在如何利用这些信号来仅从广告曝光中反向工程私人属性。我们引入了一种新颖的管道，利用LLM作为对抗推理引擎来执行自然语言分析。将这种方法应用于包含从891名用户收集的超过435，000个广告印象的纵向数据集，我们进行了一项大规模研究，以评估从被动在线广告观察中推断私人属性的可行性和精确性。我们的结果表明，现成的LLM可以准确地重建复杂的用户私人属性，包括政党偏好、就业状况和教育水平，始终优于基于人口普查的强大先验，并匹配或超过人类社会认知，同时运营成本仅为人类所需的一小部分（223美元\倍）和时间（52美元\倍）。至关重要的是，即使在短的观察窗口内，可操作的分析也是可行的，这表明长期跟踪并不是成功攻击的先决条件。这些发现提供了第一个经验证据，证明广告流可以充当高保真数字足迹，实现从本质上绕过当前平台保障措施的平台外分析，凸显了广告生态系统中的系统性漏洞以及生成性人工智能时代对负责任的网络人工智能治理的迫切需要。该代码可在https://github.com/Breezelled/when-ads-become-profiles上获取。



## **40. Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking**

你的还是我的？针对神经音频水印的覆盖攻击 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.05835v2) [paper-pdf](https://arxiv.org/pdf/2509.05835v2)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.

摘要: 随着生成音频模型的迅速发展，人工智能生成的音频越来越引发人们对版权侵权和错误信息传播的担忧。音频水印作为一种主动防御，可以将秘密消息嵌入音频中，以实现版权保护和源验证。然而，目前的神经音频水印方法主要关注水印的不可感知性和鲁棒性，而忽视了其对安全攻击的脆弱性。在本文中，我们开发了一种简单但强大的攻击：MIDI攻击，用伪造的水印覆盖合法的音频水印，并使原始的合法水印无法检测。根据对手拥有的音频水印信息，我们提出了三种类型的MIDI攻击，即，白盒、灰盒和黑匣子攻击。我们还彻底评估了对最先进的神经音频水印方法提出的攻击。实验结果表明，提出的MIDI攻击可以有效地破坏各种设置中的现有水印方案，并达到近100%的攻击成功率。提出的MIDI攻击的实用性和有效性暴露了现有神经音频水印系统中的安全缺陷，强调了在未来音频水印设计中增强安全性的必要性。



## **41. Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model**

分析二进制等PDF：通过中间表示和语言模型进行对抗稳健的PDF恶意软件分析 cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.17162v2) [paper-pdf](https://arxiv.org/pdf/2506.17162v2)

**Authors**: Side Liu, Jiang Ming, Guodong Zhou, Xinyi Liu, Jianming Fu, Guojun Peng

**Abstract**: Malicious PDF files have emerged as a persistent threat and become a popular attack vector in web-based attacks. While machine learning-based PDF malware classifiers have shown promise, these classifiers are often susceptible to adversarial attacks, undermining their reliability. To address this issue, recent studies have aimed to enhance the robustness of PDF classifiers. Despite these efforts, the feature engineering underlying these studies remains outdated. Consequently, even with the application of cutting-edge machine learning techniques, these approaches fail to fundamentally resolve the issue of feature instability.   To tackle this, we propose a novel approach for PDF feature extraction and PDF malware detection. We introduce the PDFObj IR (PDF Object Intermediate Representation), an assembly-like language framework for PDF objects, from which we extract semantic features using a pretrained language model. Additionally, we construct an Object Reference Graph to capture structural features, drawing inspiration from program analysis. This dual approach enables us to analyze and detect PDF malware based on both semantic and structural features. Experimental results demonstrate that our proposed classifier achieves strong adversarial robustness while maintaining an exceptionally low false positive rate of only 0.07% on baseline dataset compared to state-of-the-art PDF malware classifiers.

摘要: 恶意PDF文件已经成为一种持续的威胁，并成为基于Web的攻击中流行的攻击向量。虽然基于机器学习的PDF恶意软件分类器已经显示出前景，但这些分类器通常容易受到对抗性攻击，从而破坏了它们的可靠性。为了解决这个问题，最近的研究旨在提高PDF分类器的鲁棒性。尽管有这些努力，这些研究背后的特征工程仍然过时。因此，即使应用尖端的机器学习技术，这些方法也无法从根本上解决特征不稳定性问题。   为了解决这个问题，我们提出了一种新颖的PDF特征提取和PDF恶意软件检测方法。我们引入了PDFObj IR（PDF对象中间表示），这是一种用于PDF对象的类似汇编的语言框架，我们使用预先训练的语言模型从中提取语义特征。此外，我们还构建了一个对象引用图来捕获结构特征，从程序分析中汲取灵感。这种双重方法使我们能够根据语义和结构特征分析和检测PDF恶意软件。实验结果表明，我们提出的分类器实现了强大的对抗鲁棒性，同时保持了异常低的误报率只有0.07%的基线数据集相比，最先进的PDF恶意软件分类。



## **42. SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models**

SafeGenes：评估基因组基础模型的对抗稳健性 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2506.00821v2) [paper-pdf](https://arxiv.org/pdf/2506.00821v2)

**Authors**: Huixin Zhan, Clovis Barbour, Jason H. Moore

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks induced severe degradation in MLM-based shallow architectures such as ProteinBERT, while still producing substantial failure modes even in high-capacity foundation models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction.

摘要: 基因组基础模型（GFM），如进化尺度模型（ESM），已经证明在变异效应预测方面取得了显着的成功。然而，它们的对抗性鲁棒性在很大程度上仍未被探索。为了解决这一差距，我们提出了SafeGenes：一个用于基因组基础模型安全分析的框架，利用对抗性攻击来评估对工程化的几乎相同的对抗性基因和嵌入空间操作的鲁棒性。在这项研究中，我们使用两种方法评估GFM的对抗性漏洞：快速梯度符号方法（FGSM）和软提示攻击。FGSM对输入序列引入了最小的扰动，而软提示攻击优化了连续嵌入，以在不修改输入令牌的情况下操纵模型预测。通过结合这些技术，SafeGenes提供了一个全面的评估GFM对对抗性操纵的敏感性。有针对性的软提示攻击导致基于MLM的浅层架构（如ProteinBERT）严重退化，同时即使在高容量的基础模型（如ESM 1b和ESM 1v）中仍会产生大量故障模式。这些发现暴露了当前基础模型中的关键漏洞，为提高其在高风险基因组应用（如变异效应预测）中的安全性和鲁棒性开辟了新的研究方向。



## **43. Bant: Byzantine Antidote via Trial Function and Trust Scores**

Bant：通过试用功能和信任分数的拜占庭解药 cs.LG

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2505.07614v5) [paper-pdf](https://arxiv.org/pdf/2505.07614v5)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structures remain vulnerable to malicious influences. In this paper, we address a specific threat: Byzantine attacks, wherein compromised clients inject adversarial updates to derail global convergence. We combine the concept of trust scores with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing operation even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods such as Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both public datasets and private ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to the aforementioned practical setups. The convergence guaranties of our methods are comparable to those of classical algorithms developed without Byzantine interference.

摘要: 机器学习的最新进展提高了性能，同时也增加了计算需求。虽然联邦和分布式设置可以解决这些问题，但它们的结构仍然容易受到恶意影响。在本文中，我们解决了一个特定的威胁：拜占庭攻击，其中受影响的客户端注入对抗性更新以破坏全球融合。我们将信任分数的概念与尝试函数方法相结合，以动态过滤离群值。我们的方法解决了以前方法的关键局限性，即使在拜占庭节点占多数的情况下也允许运行。此外，我们的算法适用于Adam和RMSProp等广泛使用的缩放方法，以及实际场景，包括本地训练和部分参与。我们通过对公共数据集和从医疗机构收集的私人心电图数据进行广泛的实验来验证我们方法的稳健性。此外，我们还对算法及其对上述实际设置的扩展进行了广泛的理论分析。我们方法的收敛性与在没有拜占庭干扰的情况下开发的经典算法的收敛性相当。



## **44. Bones of Contention: Exploring Query-Efficient Attacks against Skeleton Recognition Systems**

争夺之骨：探索针对骨架识别系统的查询高效攻击 cs.CR

Accepted to IEEE Transactions on Information Forensics and Security (TIFS)

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2501.16843v2) [paper-pdf](https://arxiv.org/pdf/2501.16843v2)

**Authors**: Yuxin Cao, Kai Ye, Derui Wang, Minhui Xue, Hao Ge, Chenxiong Qian, Jin Song Dong

**Abstract**: Skeleton action recognition models have secured more attention than video-based ones in various applications due to privacy preservation and lower storage requirements. Skeleton data are typically transmitted to cloud servers for action recognition, with results returned to clients via Apps/APIs. However, the vulnerability of skeletal models against adversarial perturbations gradually reveals the unreliability of these systems. Existing black-box attacks all operate in a decision-based manner, resulting in numerous queries that hinder efficiency and feasibility in real-world applications. Moreover, all attacks off the shelf focus on only restricted perturbations, while ignoring model weaknesses when encountered with non-semantic perturbations. In this paper, we propose two query-effIcient Skeletal Adversarial AttaCks, ISAAC-K and ISAAC-N. As a black-box attack, ISAAC-K utilizes Grad-CAM in a surrogate model to extract key joints where minor sparse perturbations are then added to fool the classifier. To guarantee natural adversarial motions, we introduce constraints of both bone length and temporal consistency. ISAAC-K finds stronger adversarial examples on the $\ell_\infty$ norm, which can encompass those on other norms. Exhaustive experiments substantiate that ISAAC-K can uplift the attack efficiency of the perturbations under 10 skeletal models. Additionally, as a byproduct, ISAAC-N fools the classifier by replacing skeletons unrelated to the action. We surprisingly find that skeletal models are vulnerable to large perturbations where the part-wise non-semantic joints are just replaced, leading to a query-free no-box attack without any prior knowledge. Based on that, four adaptive defenses are eventually proposed to improve the robustness of skeleton recognition models.

摘要: 由于隐私保护和较低的存储要求，骨架动作识别模型在各种应用中比基于视频的动作识别模型受到了更多关注。骨架数据通常传输到云服务器进行动作识别，结果通过应用程序/API返回给客户端。然而，骨架模型对对抗性扰动的脆弱性逐渐揭示了这些系统的不可靠性。现有的黑匣子攻击都以基于决策的方式运行，从而导致大量查询，阻碍了现实世界应用程序的效率和可行性。此外，所有现成的攻击都只关注有限的扰动，而在遇到非语义扰动时忽略了模型的弱点。在本文中，我们提出了两种查询效率高的Skopper对抗AttaCks，ISAAC-K和ISAAC-N。作为黑匣子攻击，ISAAC-K在代理模型中利用Grad-CAM来提取关键关节，然后添加微小的稀疏扰动来欺骗分类器。为了保证自然的对抗运动，我们引入了骨骼长度和时间一致性的限制。ISAAC-K在$\ell_\infty$规范上找到了更强的对抗性例子，其中可以涵盖其他规范上的例子。详尽的实验证实ISAAC-K可以在10个骨架模型下提高扰动的攻击效率。此外，作为副产品，ISAAC-N通过替换与动作无关的骨架来愚弄分类器。我们惊讶地发现，骨架模型容易受到大的扰动，其中部分非语义关节刚刚被替换，从而导致在没有任何先验知识的情况下进行无查询无框攻击。在此基础上，最终提出了四种自适应防御方法来提高骨架识别模型的鲁棒性。



## **45. Improving Graph Neural Network Training, Defense, and Hypergraph Partitioning via Adversarial Robustness Evaluation**

通过对抗稳健性评估改进图神经网络训练、防御和超图划分 cs.LG

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2412.14738v7) [paper-pdf](https://arxiv.org/pdf/2412.14738v7)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. To address this issue, this paper proposes a novel method for selecting noise-sensitive training samples from the original training set to construct a smaller yet more effective training set for model training. These samples are used to help improve the model's ability to correctly process data in noisy environments. We have evaluated our approach on three of the most classical GNN models GCN, GAT, and GraphSAGE as well as three widely used benchmark datasets: Cora, Citeseer, and PubMed. Our experiments demonstrate that the proposed method can substantially boost the training of Graph Neural Networks compared to using randomly sampled training sets of the same size from the original training set and the larger original full training set. We further proposed a robust-node based hypergraph partitioning method, an adversarial robustness based graph pruning method for GNN defenses and a related spectral edge attack method.

摘要: 图神经网络（GNN）是一种用于处理图结构数据的高效神经网络架构。与仅依赖数据特征作为输入的传统神经网络不同，GNN利用表示数据点之间关系的图结构和数据的特征矩阵来优化其特征表示。这种独特的功能使GNN能够在各种任务中实现卓越的性能。然而，它也使GNN更容易受到来自图结构和数据特征的噪音的影响，这可能会显着增加训练难度并降低其性能。为了解决这个问题，本文提出了一种新的方法，从原始训练集中选择噪声敏感的训练样本，以构建一个更小但更有效的训练集模型训练。这些样本用于帮助提高模型在嘈杂环境中正确处理数据的能力。我们已经在三个最经典的GNN模型GCN，GAT和GraphSAGE以及三个广泛使用的基准数据集：Cora，Citeseer和PubMed上评估了我们的方法。我们的实验表明，与使用来自原始训练集和较大原始完整训练集的相同大小的随机采样训练集相比，所提出的方法可以大大提高图神经网络的训练。我们进一步提出了一种基于鲁棒节点的超图划分方法、一种用于GNN防御的基于对抗鲁棒性的图修剪方法以及相关的谱边缘攻击方法。



## **46. Edge-Only Universal Adversarial Attacks in Distributed Learning**

分布式学习中的仅边通用对抗攻击 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2411.10500v2) [paper-pdf](https://arxiv.org/pdf/2411.10500v2)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems, but may also introduce new vulnerabilities to evasion attacks, often in the form of adversarial perturbations. In this work, we present a new threat model that explores the feasibility of generating universal adversarial perturbations (UAPs) when the attacker has access only to the edge portion of the model, consisting of its initial network layers. Unlike traditional attacks that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud component by manipulating key feature representations at the edge. Following the proposed threat model, we introduce both edge-only untargeted and targeted formulations of UAPs designed to control intermediate features before the split point. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part, and we compare the proposed method with classical white-box and black-box techniques, highlighting its effectiveness. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effects with edge-only knowledge, revealing intriguing behaviors across multiple networks. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.

摘要: 分布式学习框架将神经网络模型划分为多个计算节点，可以提高协作边缘云系统的效率，但也可能为规避攻击引入新的漏洞，通常以对抗性扰动的形式。在这项工作中，我们提出了一种新的威胁模型，该模型探讨了当攻击者只能访问模型的边缘部分（由其初始网络层组成）时生成普遍对抗扰动（UPC）的可行性。与需要完整模型知识的传统攻击不同，我们的方法表明，对手可以通过操纵边缘的关键特征表示来在未知云组件中引发有效的误预测。遵循拟议的威胁模型，我们引入了仅边缘非定向和定向的UAP公式，旨在控制分裂点之前的中间特征。我们在ImageNet上的结果证明了攻击对未知云部分的强大可转移性，并且我们将提出的方法与经典的白盒和黑盒技术进行了比较，强调了其有效性。此外，我们还分析了攻击者利用仅边缘知识实现有针对性的对抗效果的能力，揭示了多个网络中有趣的行为。通过在分裂推理中引入第一个具有仅边知识的对抗攻击，这项工作强调了解决部分模型访问在对抗稳健性中的重要性，鼓励了该领域的进一步研究。



## **47. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

利用一对多关系对视觉语言模型进行多模式对抗防御 cs.CV

WACV 2026 Accepted. Code available at https://github.com/CyberAgentAI/multimodal-adversarial-training

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2405.18770v5) [paper-pdf](https://arxiv.org/pdf/2405.18770v5)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives. Our code is publicly available at https://github.com/CyberAgentAI/multimodal-adversarial-training.

摘要: 预先训练的视觉语言（VL）模型非常容易受到对抗性攻击。然而，现有的防御方法主要集中在图像分类上，忽略了VL任务的两个关键方面：多模态攻击，其中图像和文本都可以被扰动，以及图像和文本的一对多关系，其中单个图像可以对应于多个文本描述，反之亦然（1：N和N：1）。这项工作是第一次探索防御策略，对多模态攻击的VL任务，而以前的VL防御方法集中在视觉鲁棒性。我们提出了多模态对抗训练（MAT），它在训练过程中将对抗扰动纳入图像和文本模态，显著优于现有的单峰防御。此外，我们发现MAT受到DL训练数据中确定性的一对一（1：1）图像-文本对的限制。为了解决这个问题，我们对利用一对多关系来增强稳健性进行了全面研究，并调查了各种增强技术。我们的分析表明，为了更有效的防御，增强的图像-文本对应该对齐、多样化，但要避免分布变化--这些条件被之前的研究所忽视。这项工作开创了针对多模式攻击的防御策略，为从优化和数据角度构建稳健的VLM提供了见解。我们的代码可在https://github.com/CyberAgentAI/multimodal-adversarial-training上公开获取。



## **48. UltraClean: A Simple Framework to Train Robust Neural Networks against Backdoor Attacks**

UltraClean：训练稳健神经网络抵御后门攻击的简单框架 cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2312.10657v2) [paper-pdf](https://arxiv.org/pdf/2312.10657v2)

**Authors**: Bingyin Zhao, Yingjie Lao

**Abstract**: Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. Adversaries can activate the injected backdoor during inference by presenting the trigger on input images. Prior defensive methods have achieved remarkable success in countering dirty-label backdoor attacks where the labels of poisoned samples are often mislabeled. However, these approaches do not work for a recent new type of backdoor -- clean-label backdoor attacks that imperceptibly modify poisoned data and hold consistent labels. More complex and powerful algorithms are demanded to defend against such stealthy attacks. In this paper, we propose UltraClean, a general framework that simplifies the identification of poisoned samples and defends against both dirty-label and clean-label backdoor attacks. Given the fact that backdoor triggers introduce adversarial noise that intensifies in feed-forward propagation, UltraClean first generates two variants of training samples using off-the-shelf denoising functions. It then measures the susceptibility of training samples leveraging the error amplification effect in DNNs, which dilates the noise difference between the original image and denoised variants. Lastly, it filters out poisoned samples based on the susceptibility to thwart the backdoor implantation. Despite its simplicity, UltraClean achieves a superior detection rate across various datasets and significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforming existing defensive methods by a large margin. Code is available at https://github.com/bxz9200/UltraClean.

摘要: 后门攻击是对深度神经网络的新威胁，深度神经网络通常通过注入有毒样本将恶意行为嵌入到受害者模型中。对手可以通过在输入图像上呈现触发器来在推理期间激活注入的后门。先前的防御方法在对抗肮脏标签后门攻击方面取得了显着的成功，其中有毒样本的标签经常被错误地贴上标签。然而，这些方法不适用于最近出现的新型后门--清除标签后门攻击，这种攻击可以在不知不觉中修改有毒数据并持有一致的标签。需要更复杂和强大的算法来抵御这种隐形攻击。在本文中，我们提出了UltraClean，一个通用的框架，简化了中毒样本的识别，并抵御脏标签和干净标签后门攻击。考虑到后门触发器会引入对抗性噪声，并在前馈传播中加剧，UltraClean首先使用现成的去噪函数生成两种训练样本变体。然后，它利用DNN中的误差放大效应来测量训练样本的敏感性，该效应扩大了原始图像和去噪变体之间的噪声差异。最后，它根据阻止后门植入的敏感性过滤出有毒样本。尽管UltraClean很简单，但它在各种数据集中实现了卓越的检测率，并显着降低了后门攻击成功率，同时在干净数据上保持了不错的模型准确性，大大优于现有的防御方法。代码可在https://github.com/bxz9200/UltraClean上获取。



## **49. GPS-Spoofing Attack Detection Mechanism for UAV Swarms**

无人机群的GPS欺骗攻击检测机制 cs.CR

8 pages, 3 figures

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2301.12766v3) [paper-pdf](https://arxiv.org/pdf/2301.12766v3)

**Authors**: Pavlo Mykytyn, Marcin Brzozowski, Zoya Dyka, Peter Langendoerfer

**Abstract**: Recently autonomous and semi-autonomous Unmanned Aerial Vehicle (UAV) swarms started to receive a lot of research interest and demand from various civil application fields. However, for successful mission execution, UAV swarms require Global navigation satellite system signals and in particular, Global Positioning System (GPS) signals for navigation. Unfortunately, civil GPS signals are unencrypted and unauthenticated, which facilitates the execution of GPS spoofing attacks. During these attacks, adversaries mimic the authentic GPS signal and broadcast it to the targeted UAV in order to change its course, and force it to land or crash. In this study, we propose a GPS spoofing detection mechanism capable of detecting single-transmitter and multi-transmitter GPS spoofing attacks to prevent the outcomes mentioned above. Our detection mechanism is based on comparing the distance between each two swarm members calculated from their GPS coordinates to the distance acquired from Impulse Radio Ultra-Wideband ranging between the same swarm members. If the difference in distances is larger than a chosen threshold the GPS spoofing attack is declared detected.

摘要: 近年来，自主和半自主无人机（UF）群体开始受到各个民用应用领域的大量研究兴趣和需求。然而，为了成功执行任务，无人机群需要全球导航卫星系统信号，特别是全球定位系统（GPS）信号进行导航。不幸的是，民用GPS信号未经加密且未经验证，这有利于执行GPS欺骗攻击。在这些攻击过程中，对手模仿真实的GPS信号并将其广播给目标无人机，以改变其航线，并迫使其着陆或坠毁。在这项研究中，我们提出了一种能够检测单发射机和多发射机GPS欺骗攻击的GPS欺骗检测机制，以防止上述结果。我们的检测机制基于将根据GPS坐标计算的每两个群体成员之间的距离与从脉冲无线电超宽带测量相同群体成员之间的距离进行比较。如果距离差大于选定的阈值，则宣布检测到GPS欺骗攻击。



## **50. A review of mechanistic and data-driven models of terrorism and radicalization**

恐怖主义和激进化的机械和数据驱动模型回顾 physics.soc-ph

80 pages, 17 figures

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/1903.08485v3) [paper-pdf](https://arxiv.org/pdf/1903.08485v3)

**Authors**: Yao-li Chuang, Maria R. D'Orsogna

**Abstract**: The rapid spread of radical ideologies in recent years has led to a worldwide string of terrorist attacks. Understanding how extremist tendencies germinate, develop, and drive individuals to action is important from a cultural standpoint, but also to help formulate response and prevention strategies. Demographic studies, interviews with radicalized subjects, analysis of terrorist databases, reveal that the path to radicalization occurs along progressive steps, where age, social context and peer-to-peer exchange of extremist ideas play major roles. Furthermore, the advent of social media has offered new channels of communication, facilitated recruitment, and hastened the leap from mild discontent to unbridled fanaticism. While a complete sociological understanding of the processes and circumstances that lead to full-fledged extremism is still lacking, quantitative approaches, using modeling and data analyses, can offer useful insight. We review some approaches from statistical mechanics, applied mathematics, data science, that can help describe and understand radicalization and terrorist activity. Specifically, we focus on compartment models of populations harboring extremist views, continuous time models for age-structured radical populations, radicalization as social contagion processes on lattices and social networks, adversarial evolutionary games coupling terrorists and counter-terrorism agents, and point processes to study the spatiotemporal clustering of terrorist events. We also present recent applications of machine learning methods on open-source terrorism databases. Finally, we discuss the role of institutional intervention and the stages at which de-radicalization strategies might be most effective.

摘要: 近年来，激进意识形态的迅速传播导致了一系列全球恐怖袭击。从文化的角度来看，了解极端主义倾向如何萌芽、发展和驱使个人采取行动非常重要，而且有助于制定应对和预防策略。人口研究、对激进对象的采访、对恐怖分子数据库的分析表明，激进化的道路是沿着渐进的步骤发生的，其中年龄、社会背景和极端主义思想的点对点交流发挥着重要作用。此外，社交媒体的出现提供了新的沟通渠道，促进了招聘，并加速了从轻微不满到肆无忌惮的狂热的飞跃。虽然仍然缺乏对导致全面极端主义的过程和环境的完整社会学理解，但使用建模和数据分析的定量方法可以提供有用的见解。我们回顾了统计力学、应用数学、数据科学的一些方法，这些方法可以帮助描述和理解激进化和恐怖活动。具体来说，我们重点关注持有极端主义观点的人群的隔间模型、年龄结构激进人群的连续时间模型、作为格子和社交网络上社会传染过程的激进化、将恐怖分子和反恐特工相结合的对抗进化游戏，以及研究恐怖事件时空聚集的点过程。我们还介绍了机器学习方法在开源恐怖主义数据库上的最新应用。最后，我们讨论了制度干预的作用以及去激进化策略可能最有效的阶段。



