# Latest Large Language Model Attack Papers
**update at 2025-03-31 15:45:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Training Large Language Models for Advanced Typosquatting Detection**

训练大型语言模型以进行高级排字检测 cs.CR

6 pages, 1 figure

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22406v1) [paper-pdf](http://arxiv.org/pdf/2503.22406v1)

**Authors**: Jackson Welch

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection.

摘要: Typosquating是一种长期存在的网络威胁，它利用键入URL时的人为错误来欺骗用户、分发恶意软件和进行网络钓鱼攻击。随着域名和新的顶级域名(TLD)的激增，类型匹配技术变得更加复杂，给个人、企业和国家网络安全基础设施带来了重大风险。传统的检测方法主要集中在众所周知的模拟模式上，在识别更复杂的攻击方面存在空白。这项研究介绍了一种利用大语言模型(LLM)来增强类型匹配检测的新方法。通过在字符级转换和基于模式的启发式方法而不是特定于领域的数据上训练LLM，开发了一种更具适应性和弹性的检测机制。实验结果表明，在适当微调的情况下，Phi-414B模型优于其他测试模型，仅用几千个训练样本就能达到98%的准确率。这项研究突出了LLMS在网络安全应用中的潜力，特别是在减轻基于域的欺骗策略方面，并为优化威胁检测的机器学习策略提供了见解。



## **2. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

单图像去学习：多模式大型语言模型中的高效机器去学习 cs.CV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2405.12523v3) [paper-pdf](http://arxiv.org/pdf/2405.12523v3)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, Fan Liu

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.

摘要: 机器遗忘通过删除编码在机器学习模型中的私人或敏感信息，使个人有被遗忘的权利。然而，MU是否能有效地应用于多通道大语言模型，特别是在忘记概念的泄露的视觉数据的情况下，仍然是不确定的。为了克服这一挑战，我们提出了一种有效的方法，单图像忘却学习(SIU)，通过对单个关联图像进行几个步骤的微调来消除对概念的视觉识别。SIU包括两个关键方面：(I)构建多方面的微调数据。我们引入了四个目标，在此基础上构建了精调的数据，以使概念被遗忘；(Ii)联合训练损失。为了同步忘记对概念的视觉识别，同时保持MLLS的实用性，我们通过一种新颖的双屏蔽KL-发散损失和交叉熵损失来微调MLLMS。除了我们的方法之外，我们还建立了MLLMS中MU的一个新的基准MMUBENCH，并引入了一组用于评估它的度量。在MMUBENCH上的实验结果表明，SIU的性能完全优于现有方法。此外，我们惊讶地发现，SIU可以避免入侵性的成员推理攻击和越狱攻击。据我们所知，我们是第一个探索MLLMS中的MU的人。我们将在不久的将来发布代码和基准测试。



## **3. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

AnyAttack：走向对视觉语言模型的大规模自我监督对抗攻击 cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.

摘要: 由于其多模式功能，视觉语言模型（VLM）在现实世界场景中发现了许多有影响力的应用程序。然而，最近的研究表明，VLM很容易受到基于图像的对抗攻击。传统的有针对性的对抗攻击需要特定的目标和标签，从而限制了其现实世界的影响。我们提出AnyAttack，这是一个自我监督的框架，通过新颖的基础模型方法超越了传统攻击的局限性。通过在无需标签监督的情况下在海量LAION-400 M数据集上进行预训练，AnyAttack实现了前所未有的灵活性-使任何图像都能转化为攻击载体，目标是不同的LMA之间的任何所需输出。这种方法从根本上改变了威胁格局，使对抗能力以前所未有的规模获得。我们对五种开源VLM（CLIP、BLIP、BLIP 2、INSTBLIP和MiniGPT-4）进行了广泛验证，证明了AnyAttack在各种多模式任务中的有效性。最令人担忧的是，AnyAttack无缝传输到包括Google Gemini、Claude Sonnet、Microsoft Copilot和OpenAI GPT在内的商业系统，揭示了一个需要立即关注的系统性漏洞。



## **4. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

一脚踏进门：LLC的多次越狱 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.

摘要: 随着大型语言模型越来越多地融入现实世界的应用程序中，确保人工智能的安全至关重要。一个关键的挑战是越狱，敌意提示绕过内置的保护措施，导致有害的不允许输出。受心理学进门原则的启发，我们引入了FITD，这是一种新颖的多转弯越狱方法，它利用了这样一种现象，即较小的初始承诺降低了对更重大或更不道德的违法行为的抵抗力。我们的方法通过中间桥提示逐步升级用户查询的恶意意图，并使模型本身的响应保持一致，以诱导有毒响应。在两个越狱基准上的广泛实验结果表明，FITD在七个广泛使用的模型上实现了94%的平均攻击成功率，性能优于现有的最先进方法。此外，我们还提供了对LLM自我腐败的深入分析，强调了当前调整策略中的漏洞，并强调了多轮交互中固有的风险。代码可在https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.上获得



## **5. Debate-Driven Multi-Agent LLMs for Phishing Email Detection**

用于网络钓鱼电子邮件检测的辩论驱动的多代理LLM cs.MA

Accepted to the 13th International Symposium on Digital Forensics and  Security (ISDFS 2025)

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22038v1) [paper-pdf](http://arxiv.org/pdf/2503.22038v1)

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies.

摘要: 网络钓鱼攻击仍然是一个严重的网络安全威胁。攻击者不断改进他们的方法，使钓鱼电子邮件更难被发现。传统的检测方法，包括基于规则的系统和有监督的机器学习模型，要么依赖于预定义的模式，如黑名单，只需稍加修改即可绕过，要么需要大量数据集进行训练，仍然可能产生假阳性和假阴性。在这项工作中，我们提出了一种多智能体大语言模型(LLM)提示技术，模拟智能体之间的辩论来检测电子邮件上呈现的内容是否为网络钓鱼。我们的方法使用两个LLM代理来提出支持或反对分类任务的论点，由一个判断代理根据提供的推理质量来判决最终裁决。这种辩论机制使模型能够批判性地分析文本中的上下文线索和欺骗模式，从而提高了分类精度。在多个钓鱼电子邮件数据集上对所提出的框架进行了评估，结果表明混合代理配置的性能始终优于同构配置。结果还表明，辩论结构本身足以产生准确的决定，而不需要额外的提示策略。



## **6. Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base**

基于特征排序知识库的ODLLM智能IoT攻击检测设计 cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21674v1) [paper-pdf](http://arxiv.org/pdf/2503.21674v1)

**Authors**: Satvik Verma, Qun Wang, E. Wes Bethel

**Abstract**: The widespread adoption of Internet of Things (IoT) devices has introduced significant cybersecurity challenges, particularly with the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks. Traditional machine learning (ML) techniques often fall short in detecting such attacks due to the complexity of blended and evolving patterns. To address this, we propose a novel framework leveraging On-Device Large Language Models (ODLLMs) augmented with fine-tuning and knowledge base (KB) integration for intelligent IoT network attack detection. By implementing feature ranking techniques and constructing both long and short KBs tailored to model capacities, the proposed framework ensures efficient and accurate detection of DDoS attacks while overcoming computational and privacy limitations. Simulation results demonstrate that the optimized framework achieves superior accuracy across diverse attack types, especially when using compact models in edge computing environments. This work provides a scalable and secure solution for real-time IoT security, advancing the applicability of edge intelligence in cybersecurity.

摘要: 物联网(IoT)设备的广泛采用带来了重大的网络安全挑战，特别是随着分布式拒绝服务(DDoS)攻击的频率和复杂性不断增加。由于混合和演化模式的复杂性，传统的机器学习(ML)技术往往无法检测到此类攻击。为了解决这一问题，我们提出了一种新的框架，该框架利用设备上的大语言模型(ODLLMS)，并结合微调和知识库(KB)集成，用于智能物联网网络攻击检测。通过实现特征排序技术和根据模型能力构建长和短知识库，该框架在克服计算和隐私限制的同时，确保了对DDoS攻击的高效和准确检测。仿真结果表明，优化后的框架对于不同的攻击类型具有较高的准确率，特别是在边缘计算环境中使用紧凑模型的情况下。这项工作为实时物联网安全提供了可扩展的安全解决方案，提升了边缘智能在网络安全中的适用性。



## **7. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

CleanGen：减轻大型语言模型中生成任务的后门攻击 cs.AI

This paper is presented at EMNLP 2024

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2406.12257v3) [paper-pdf](http://arxiv.org/pdf/2406.12257v3)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.

摘要: 大型语言模型（LLM）在生成任务中的出色性能使从业者能够利用公开可用的模型来支持自定义应用程序，例如聊天机器人和虚拟助理。然而，用于训练或微调这些LLM的数据通常是不公开的，这使得攻击者能够破坏数据并将后门注入模型。在本文中，我们开发了一种新型的推理时间防御，名为CleANGER，以减轻对LLM中生成任务的后门攻击。Cleangen是一种轻量级且有效的解码策略，与最先进的（SOTA）LLM兼容。我们对Cleangen的见解是，与其他LLM相比，后门LLM为代表攻击者所需内容的令牌分配了明显更高的概率。令牌概率的这些差异使CleANGER能够识别攻击者青睐的可疑令牌，并用未被同一攻击者泄露的另一个LLM生成的令牌替换它们，从而避免生成攻击者想要的内容。我们针对五种SOTA后门攻击评估了Cleangen。我们的结果表明，与五种SOTA基线防御相比，对于所有五种后门攻击，CleANGER的攻击成功率（ASB）更低。此外，部署Cleangen的LLM在以最小的额外计算负担为良性用户查询提供服务时保持其响应的有用性。



## **8. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

利用思想链元数据进行任务路由和对抗提示检测 cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.

摘要: 在这项工作中，我们提出了一种名为“思考数量”（NofT）的指标，以确定预提示任务的难度并支持生产环境中的大型语言模型（LLM）。通过根据想法数量设置阈值，该指标可以辨别提示的难度并支持更有效的提示路由。当通过具有17亿、70亿和140亿参数的量化、提炼版本的Deepseek从MathDirect数据集中路由提示时，延迟可降低2%。此外，该指标可用于高效检测提示注射攻击中使用的对抗提示。思维数量可以通知分类器，在对抗性提示检测中达到95%的准确率。我们使用的实验和数据集可以在我们的GitHub页面上找到：https://github.com/rymarinelli/Number_Of_Thoughts/tree/main。



## **9. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

用有影响力的代币欺骗猎犬：一种有效的黑匣子库中毒攻击 cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.

摘要: 检索增强生成（RAG）系统通过整合外部知识来增强大型语言模型，解决过时的内部知识和幻觉等问题。然而，它们对外部知识库的依赖使它们容易受到文集中毒攻击，其中可以注入对抗性段落来操纵检索结果。用于制作此类段落的现有方法，例如随机令牌替换或训练倒置模型，通常速度缓慢且计算昂贵，需要访问检索器的梯度或大量计算资源。为了解决这些限制，我们提出了动态重要引导遗传算法（DIGA），这是一种有效的黑匣子方法，利用检索器的两个关键属性：对代币顺序的不敏感和对有影响力代币的偏见。通过关注这些特征，DIGA动态地调整其遗传操作，以生成有效的对抗通道，并显着减少时间和内存使用。我们的实验评估表明，与现有方法相比，DIGA实现了更高的效率和可扩展性，同时在多个数据集上保持了可比或更好的攻击成功率。



## **10. M-LLM Based Video Frame Selection for Efficient Video Understanding**

基于M-LLM的视频帧选择以实现高效的视频理解 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.19680v2) [paper-pdf](http://arxiv.org/pdf/2502.19680v2)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.

摘要: 多模式大型语言模型（M-LLM）的最新进展在视频推理方面显示出有希望的结果。流行的多模式大型语言模型（M-LLM）框架通常应用朴素均匀采样来减少输入M-LLM的视频帧数量，特别是对于长上下文视频。然而，它可能会在视频的某些时段失去关键的上下文，因此下游M-LLM可能没有足够的视觉信息来回答问题。为了解决这个痛点，我们提出了一种轻量级的基于M-LLM的帧选择方法，该方法自适应地选择与用户查询更相关的帧。为了训练提出的帧选择器，我们引入了两个监督信号（i）空间信号，其中通过提示M-LLM来评分单个帧重要性;（ii）时间信号，其中通过提示大型语言模型（LLM）来选择多个帧。所有候选帧的字幕。然后，通过冻结的下游视频M-LLM消化所选的帧，以进行视觉推理和问答。经验结果表明，提出的M-LLM视频帧选择器提高了跨媒体（ActivityNet、NExT-QA）和长（EgoSCA、LongVideoBench）上下文视频问答基准的各种下游视频大型语言模型（video-LLM）的性能。



## **11. Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models**

越狱大型语言模型中具有说服技巧的迭代绘图 cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20320v1) [paper-pdf](http://arxiv.org/pdf/2503.20320v1)

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt.

摘要: 大型语言模型（LLM）旨在在其响应中与人类价值观保持一致。这项研究通过迭代提示技术来利用LLM，其中每个提示都经过多次迭代系统地修改和细化，以逐步增强其越狱攻击的有效性。该技术涉及分析LLM的响应模式，包括GPT-3.5、GPT-4、LLaMa 2、Vicuna和ChatGLM，使我们能够调整和优化提示以规避LLM的道德和安全限制。说服策略增强了及时的有效性，同时与恶意意图保持一致。我们的结果表明，随着攻击提示变得更加精确，攻击成功率（ASB）也会增加，GPT 4和ChatGLM的最高ASB为90%，LLaMa 2的最低ASB为68%。我们的技术在ASB中优于基线技术（PAIR和PAP），并表现出与GCG和ArtPrompt相当的性能。



## **12. sudo rm -rf agentic_security**

sudo rm -ref agentic_secure cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20279v1) [paper-pdf](http://arxiv.org/pdf/2503.20279v1)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs.

摘要: 大型语言模型(LLM)越来越多地被部署为计算机使用代理，在真实的桌面或Web环境中自主执行任务。虽然这一演变极大地扩展了人类的实际用例，但也造成了严重的安全风险。我们提出了SUDO(基于屏幕的通用脱毒2Tox攻击)，这是一种新的攻击框架，它系统地绕过了商业计算机使用代理(如Claude计算机使用)中拒绝训练的保护措施。核心机制Detox2Tox通过解毒将有害的请求(代理最初拒绝的)转换为看似良性的请求，从高级视觉语言模型(VLM)获得详细说明，然后在执行前通过中毒重新引入恶意内容。与传统的越狱不同，数独基于内置的拒绝反馈反复改进其攻击，使其对强大的策略过滤器越来越有效。在跨越50个真实世界任务和多个最先进的VLM的广泛测试中，SUDO实现了24%(未经改进)的赤裸裸攻击成功率，以及克劳德计算机使用的高达41%(通过迭代优化)。通过揭示这些漏洞并展示在真实计算环境中利用它们是多么容易，本白皮书强调了对强大的、上下文感知的安全保护的迫切需求。警告：本文包括有害或攻击性模型输出。



## **13. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

大视觉语言模型的自我监督学习视觉编码器中的秘密后门攻击 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.18290v3) [paper-pdf](http://arxiv.org/pdf/2502.18290v3)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

摘要: 自监督学习(SSL)视觉编码者学习高质量的图像表征，因此成为开发大型视觉语言模型(LVLMS)视觉通道的重要组成部分。由于培训这类编码器的成本很高，预先训练的编码器被广泛共享并部署到许多安全关键或具有社会意义的LVLM中。在这种实际情况下，我们揭示了一种新的后门威胁，即仅仅通过损害视觉编码器就可以在这些LVLM中诱导出显著的视觉幻觉。由于这些编码器的共享和重用，许多下游的LVLM可能会继承编码器的后门行为，导致广泛的后门。在这项工作中，我们提出了BadVision，这是第一个通过新颖的触发优化和后门学习技术来利用LVLM的SSL视觉编码器中的漏洞的方法。我们在八个基准测试中评估了BadVision在两种类型的SSL编码器和LVLM上的性能。结果表明，BadVision在保持隐蔽性的同时，以99%以上的攻击成功率有效地驱动了LVLM进入攻击者选择的幻觉，导致了77.6%的相对视觉理解错误。SOTA后门检测方法不能有效检测到我们的攻击。



## **14. TeleLoRA: Teleporting Model-Specific Alignment Across LLMs**

TeleLoRA：LLM之间远程传输模型特定的一致 cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20228v1) [paper-pdf](http://arxiv.org/pdf/2503.20228v1)

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.

摘要: 缓解大型语言模型（LLM）中的特洛伊木马是对齐数据特定于LLM的众多任务之一，因为不同的LLM具有不同的特洛伊木马触发器和要删除的触发行为。在本文中，我们介绍了TeleLoRA（远程传输低等级自适应），这是一种新颖的框架，可以在多个LLM之间协同特定于模型的对齐数据，以便在没有对齐数据的情况下对不可见的LLM实现零触发特洛伊木马缓解。TeleLoRA通过利用多个LLM之间的本地激活信息来学习LoRA适配器权重的统一生成器。该生成器被设计为排列对称，以便在具有不同架构和大小的模型之间进行概括。我们优化了模型设计以提高内存效率，使以最少的计算资源使用大规模LLM进行学习成为可能。LLM特洛伊木马缓解基准测试的实验表明，TeleLoRA有效降低了攻击成功率，同时保持了模型的良性性能。



## **15. Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy**

扮演傻瓜：越狱LLC和具有分销外策略的多模式LLC cs.CR

Accepted at CVPR2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20823v1) [paper-pdf](http://arxiv.org/pdf/2503.20823v1)

**Authors**: Joonhyun Jeong, Seyun Bae, Yeonsung Jung, Jaeryong Hwang, Eunho Yang

**Abstract**: Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety alignment. Experiments across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and o1 with high attack success rate, which previous attack approaches have consistently struggled to jailbreak. Code is available at https://github.com/naver-ai/JOOD.

摘要: 尽管大型语言模型(LLM)和多模式LLM(MLLM)具有惊人的通用性，可以跨语言和视觉任务进行概括，但LLM和MLLM在越狱方面表现出脆弱性，当接触到有害或敏感的输入时，会生成破坏安全、道德和偏见标准的文本输出。随着最近通过人类反馈调整偏好来促进安全匹配，LLM和MLLM已经配备了安全护栏，以对有害输入做出安全、合乎道德和公平的反应。然而，尽管安全调整具有重要意义，但对漏洞的研究在很大程度上仍未得到充分探索。在本文中，我们调查了安全对齐的未知漏洞，检查了其一致地为分布外(OOD)提供安全保证的能力-使可能属于对齐的数据分布之外的有害输入。我们的主要观察是，面向对象的有害输入极大地增加了模型的不确定性，以识别输入中的恶意意图，从而导致更高的越狱机会。利用这一漏洞，我们提出了Jood，一个新的越狱框架，通过对超出安全对齐的输入进行面向对象设计。我们探索了各种现成的视觉和文本转换技术，以实现有害输入的OOD。值得注意的是，我们观察到，即使是简单的基于混合的技术，如图像混合，也被证明在增加模型的不确定性方面非常有效，从而有助于绕过安全对齐。在各种越狱场景中的实验表明，Jood有效地越狱了最近拥有专利的LLM和MLLM，如GPT-4和O1，攻击成功率很高，而以前的攻击方法一直难以越狱。代码可在https://github.com/naver-ai/JOOD.上找到



## **16. Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection**

从LLM到出处分析的知识转移：APT检测的语义增强方法 cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.18316v2) [paper-pdf](http://arxiv.org/pdf/2503.18316v2)

**Authors**: Fei Zuo, Junghwan Rhee, Yung Ryn Choe

**Abstract**: Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.

摘要: 高级持续威胁（APT）已在各个行业造成重大损失，包括敏感数据被盗和系统完整性受损。随着攻击技术变得越来越复杂和隐蔽，网络防御者和攻击者之间的军备竞赛继续加剧。大型语言模型（LLM）的革命性影响为包括网络安全在内的各个领域开辟了众多机会。一个有趣的问题出现了：LLM中嵌入的广泛知识能否用于来源分析，并在识别之前未知的恶意事件方面发挥积极作用？为了更深入地了解这个问题，我们提出了一种新的策略，用于在基于来源的威胁检测中利用LLM。在我们的设计中，最先进的LLM利用他们对系统调用、软件身份和对应用程序执行上下文的高级理解，提供了出处数据解释的更多细节。进一步利用先进的上下文嵌入能力来捕获事件描述的丰富语义。我们全面检查了所得嵌入的质量，事实证明它们提供了有希望的途径。随后，基于这些嵌入构建的机器学习模型在现实世界数据上表现出出色的性能。在我们的评估中，监督式威胁检测的准确率为99.0%，半监督式异常检测的准确率为96.9%。



## **17. Inducing Personality in LLM-Based Honeypot Agents: Measuring the Effect on Human-Like Agenda Generation**

基于LLM的蜜罐代理中的人格诱导：衡量对类人议程生成的影响 cs.AI

11 pages, 1 figure, 6 tables. Accepted to NLPAICS 2024

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19752v1) [paper-pdf](http://arxiv.org/pdf/2503.19752v1)

**Authors**: Lewis Newsham, Ryan Hyland, Daniel Prince

**Abstract**: This paper presents SANDMAN, an architecture for cyber deception that leverages Language Agents to emulate convincing human simulacra. Our 'Deceptive Agents' serve as advanced cyber decoys, designed for high-fidelity engagement with attackers by extending the observation period of attack behaviours. Through experimentation, measurement, and analysis, we demonstrate how a prompt schema based on the five-factor model of personality systematically induces distinct 'personalities' in Large Language Models. Our results highlight the feasibility of persona-driven Language Agents for generating diverse, realistic behaviours, ultimately improving cyber deception strategies.

摘要: 本文介绍了SAANDMAN，这是一种网络欺骗架构，它利用语言代理来模拟令人信服的人类拟像。我们的“欺骗性特工”充当高级网络诱饵，旨在通过延长攻击行为的观察期来与攻击者进行高保真接触。通过实验、测量和分析，我们展示了基于人格五因素模型的提示模式如何系统地在大型语言模型中诱导不同的“人格”。我们的结果强调了人物驱动的语言代理生成多样化、现实的行为、最终改进网络欺骗策略的可行性。



## **18. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted in ICLR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2412.03235v2) [paper-pdf](http://arxiv.org/pdf/2412.03235v2)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型(LLM)容易受到精心设计的对抗性攻击或越狱，尽管使用安全微调方法与人类的偏好保持一致，但这些攻击或越狱会导致生成令人反感的内容。虽然输入令牌空间的大维度使得找到能够越狱这些模型的敌意提示是不可避免的，但我们的目标是评估安全的微调LLM对于自然提示是否安全，这些自然提示在语义上与有毒种子提示相关，在对齐后引起安全响应。我们惊讶地发现，GPT-4等流行的对齐LLM可以使用甚至不是以越狱为目标而精心设计的幼稚提示来进行攻击。此外，我们的经验表明，给定一个种子提示引起来自未对齐模型的有毒反应，一个人可以系统地生成几个语义相关的自然提示，从而可以越狱对齐的LLM。为此，我们提出了一种反应引导问题增强方法(REG-QA)来评估安全对齐LLM对自然提示的泛化，该方法首先使用未对齐LLM(Q到A)来生成给定种子问题的几个有毒答案，然后利用LLM来生成可能产生这些答案(A到Q)的问题。有趣的是，我们发现安全微调的LLM，如GPT-40，容易从不安全的内容产生自然的越狱问题(不否认)，因此可以用于后一步(A到Q)。我们获得了相当于/好于JailBreak排行榜上领先的对抗性攻击方法的攻击成功率，同时对Smooth-LLM和同义词替换等防御措施明显更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **19. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

16 pages, 7 figures

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.21805v1) [paper-pdf](http://arxiv.org/pdf/2503.21805v1)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership.

摘要: 培训大型语言模型(LLM)是资源密集型和昂贵的，这使得知识产权(IP)保护至关重要。现有的大多数模型指纹方法都将指纹注入到LLM中以保护模型的所有权。这些方法产生的指纹对具有弱的语义相关性，缺乏正常问答对中的上下文连贯性和语义关联性。在本文中，我们提出了一种世代修订干预(GRI)攻击，可以有效地利用该漏洞来擦除指纹，突出了对更安全的模型指纹方法的需求。因此，我们提出了一种新的注入指纹范式，称为隐含指纹(IMF)。IMF构造具有强语义相关性的指纹对，将它们伪装成LLMS中的自然QA对。这确保指纹与正常模式行为一致，使它们无法区分，并且对检测和删除具有很强的抵抗力。我们在多个LLM上的实验表明，IMF在对抗条件下保持了较高的验证成功率，为保护LLM所有权提供了可靠的解决方案。



## **20. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19338v1) [paper-pdf](http://arxiv.org/pdf/2503.19338v1)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since the ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs), techniques used to determine whether a particular data point was part of a model's training set, serve as a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in traditional models, there remains a lack of systematic surveys addressing their effectiveness and implications in modern large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on their methodology and scenario and discussed the limitations in existing research. Additionally, we examine privacy concerns associated with the fine-tuning process. Finally, we provided some suggestions for future research in this direction.

摘要: 自OpenAI的ChatGPT于2022年11月上线以来，大型语言模型(LLM)的采用速度急剧加快。大型多模式模型(LMM)可以处理不同的数据类型并通过各种渠道进行交互，其最新进展已经超越了早期LMM的文本到文本的限制，吸引了研究人员和工业界的广泛关注。在LLMS和LMM开始广泛传播的同时，人们对其隐私风险的担忧也在增加。成员关系推理攻击(MIA)是一种用于确定特定数据点是否属于模型训练集的技术，是评估机器学习模型隐私漏洞的关键指标。Hu等人。表明各种机器学习算法都容易受到MIA的攻击。尽管对传统模型中的MIA进行了广泛的研究，但仍然缺乏系统的调查来研究它们在LLMS和LMM等现代大规模模型中的有效性和影响。本文系统地综述了近年来MIA抗低密度脂蛋白和低密度脂蛋白的研究。我们根据它们的方法和场景对每种攻击进行了分析和分类，并讨论了现有研究中的局限性。此外，我们还检查了与微调过程相关的隐私问题。最后，我们对未来这方面的研究提出了一些建议。



## **21. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

h4 rm3l：可组合越狱攻击合成语言 cs.CR

Accepted to the Thirteenth International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2408.04811v4) [paper-pdf](http://arxiv.org/pdf/2408.04811v4)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.

摘要: 尽管具有宝贵的能力，但最先进的（SOTA）广泛部署的大型语言模型（LLM）仍然有可能对社会造成伤害，因为其安全过滤器无效，而安全过滤器可以被称为越狱攻击的即时转换绕过。当前的LLM安全评估方法采用模板化提示和基准管道的数据集，未能涵盖足够大和多样化的越狱攻击集，导致不安全的LLM的广泛部署。最近的研究表明，新颖的越狱攻击可以通过合成来衍生;然而，越狱攻击的正式可合成表示，除了其他好处外，可以通过程序合成方法探索越狱攻击的大合成空间，以前还没有提出过。我们引入了h4 rm 3l，这是一种新颖的方法，可以通过人类可读的特定领域语言（DSA）来解决这一差距。我们的框架包括：（1）h4 rm 3l DSL，它将越狱攻击正式表达为参数化字符串转换基元的组合。(2)一个具有强盗算法的合成器，可有效生成针对目标黑匣子LLM优化的越狱攻击。(3)h4 rm 3l红色团队软件工具包采用前两个组件和与人类判断高度一致的自动有害LLM行为分类器。我们通过合成针对6个SOTA开源和专有LLM的2656个成功新型越狱攻击的数据集，并针对这些合成攻击的子集对这些模型进行基准测试来证明h4 rm 3l的功效。我们的结果表明，h4 rm 3l的合成攻击多种多样，并且比文献中现有的越狱攻击更成功，SOTA LLM上的成功率超过90%。



## **22. MIRAGE: Multimodal Immersive Reasoning and Guided Exploration for Red-Team Jailbreak Attacks**

MIARCH：红队越狱袭击的多模式沉浸式推理和引导探索 cs.CL

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19134v1) [paper-pdf](http://arxiv.org/pdf/2503.19134v1)

**Authors**: Wenhao You, Bryan Hooi, Yiwei Wang, Youke Wang, Zong Ke, Ming-Hsuan Yang, Zi Huang, Yujun Cai

**Abstract**: While safety mechanisms have significantly progressed in filtering harmful text inputs, MLLMs remain vulnerable to multimodal jailbreaks that exploit their cross-modal reasoning capabilities. We present MIRAGE, a novel multimodal jailbreak framework that exploits narrative-driven context and role immersion to circumvent safety mechanisms in Multimodal Large Language Models (MLLMs). By systematically decomposing the toxic query into environment, role, and action triplets, MIRAGE constructs a multi-turn visual storytelling sequence of images and text using Stable Diffusion, guiding the target model through an engaging detective narrative. This process progressively lowers the model's defences and subtly guides its reasoning through structured contextual cues, ultimately eliciting harmful responses. In extensive experiments on the selected datasets with six mainstream MLLMs, MIRAGE achieves state-of-the-art performance, improving attack success rates by up to 17.5% over the best baselines. Moreover, we demonstrate that role immersion and structured semantic reconstruction can activate inherent model biases, facilitating the model's spontaneous violation of ethical safeguards. These results highlight critical weaknesses in current multimodal safety mechanisms and underscore the urgent need for more robust defences against cross-modal threats.

摘要: 虽然安全机制在过滤有害文本输入方面取得了重大进展，但MLLMS仍然容易受到利用其跨模式推理能力的多模式越狱的影响。我们提出了一种新颖的多模式越狱框架--幻影，它利用叙事驱动的上下文和角色沉浸来规避多模式大语言模型(MLLMS)中的安全机制。通过系统地将有毒问题分解为环境、角色和动作三元组，幻影使用稳定扩散构建了一个由图像和文本组成的多轮视觉讲故事序列，通过引人入胜的侦探叙事引导目标模型。这个过程逐渐降低了模型的防御能力，并通过结构化的上下文线索巧妙地指导其推理，最终引发有害的反应。在使用六个主流MLLM的选定数据集上进行的广泛实验中，幻影实现了最先进的性能，在最佳基线上将攻击成功率提高了17.5%。此外，我们证明角色沉浸和结构化语义重构可以激活固有的模型偏差，促进模型自发地违反伦理保障。这些结果突出了当前多式联运安全机制的严重弱点，并强调迫切需要更有力地防御跨多式联运威胁。



## **23. Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification**

面具和模仿：对作者身份验证的战略混淆和模仿攻击 cs.CL

Accepted at NLP4DH Workshop @ NAACL 2025

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19099v1) [paper-pdf](http://arxiv.org/pdf/2503.19099v1)

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively.

摘要: 人工智能（AI）技术（例如大型语言模型（LLM））的越来越多的使用导致了各种任务的重要改进，包括文档的准确作者身份识别。然而，在LLM改进此类防御技术的同时，它们也同时为恶意行为者提供了发起新攻击载体的工具。为了应对这种安全风险，我们评估了作者身份模型（特别是作者身份验证模型）对强大的基于LLM的攻击的对抗稳健性。这些攻击包括非目标方法- \textit{authorship obfuscation}和目标方法- \textit{authorship imperation}。对于这两种攻击，目标是分别掩盖或模仿作者的写作风格，同时保留原始文本的语义。因此，我们扰乱了准确的作者身份验证模型，并分别实现了模糊和模仿攻击的最大攻击成功率92%和78%。



## **24. Defeating Prompt Injections by Design**

通过设计击败提示注射 cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18813v1) [paper-pdf](http://arxiv.org/pdf/2503.18813v1)

**Authors**: Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly deployed in agentic systems that interact with an external environment. However, LLM agents are vulnerable to prompt injection attacks when handling untrusted data. In this paper we propose CaMeL, a robust defense that creates a protective system layer around the LLM, securing it even when underlying models may be susceptible to attacks. To operate, CaMeL explicitly extracts the control and data flows from the (trusted) query; therefore, the untrusted data retrieved by the LLM can never impact the program flow. To further improve security, CaMeL relies on a notion of a capability to prevent the exfiltration of private data over unauthorized data flows. We demonstrate effectiveness of CaMeL by solving $67\%$ of tasks with provable security in AgentDojo [NeurIPS 2024], a recent agentic security benchmark.

摘要: 大型语言模型（LLM）越来越多地部署在与外部环境交互的代理系统中。然而，LLM代理在处理不受信任的数据时很容易受到提示注入攻击。在本文中，我们提出了CaMeL，这是一种强大的防御措施，可以在LLM周围创建一个保护系统层，即使底层模型可能容易受到攻击，也可以保护它。为了操作，CaMeL从（受信任）查询中显式提取控制和数据流;因此，LLM检索的不受信任数据永远不会影响程序流。为了进一步提高安全性，CaMeL依赖于防止私人数据通过未经授权的数据流泄露的能力的概念。我们通过在AgentDojo [NeurIPS 2024]（最近的AgentDojo）中解决具有可证明安全性的价值67%的任务来证明CaMeL的有效性。



## **25. MF-CLIP: Leveraging CLIP as Surrogate Models for No-box Adversarial Attacks**

MF-CLIP：利用CLIP作为无框对抗攻击的代理模型 cs.LG

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2307.06608v3) [paper-pdf](http://arxiv.org/pdf/2307.06608v3)

**Authors**: Jiaming Zhang, Lingyu Qiu, Qi Yi, Yige Li, Jitao Sang, Changsheng Xu, Dit-Yan Yeung

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial attacks poses a significant challenge to their deployment in safety-critical applications. While extensive research has addressed various attack scenarios, the no-box attack setting where adversaries have no prior knowledge, including access to training data of the target model, remains relatively underexplored despite its practical relevance. This work presents a systematic investigation into leveraging large-scale Vision-Language Models (VLMs), particularly CLIP, as surrogate models for executing no-box attacks. Our theoretical and empirical analyses reveal a key limitation in the execution of no-box attacks stemming from insufficient discriminative capabilities for direct application of vanilla CLIP as a surrogate model. To address this limitation, we propose MF-CLIP: a novel framework that enhances CLIP's effectiveness as a surrogate model through margin-aware feature space optimization. Comprehensive evaluations across diverse architectures and datasets demonstrate that MF-CLIP substantially advances the state-of-the-art in no-box attacks, surpassing existing baselines by 15.23% on standard models and achieving a 9.52% improvement on adversarially trained models. Our code will be made publicly available to facilitate reproducibility and future research in this direction.

摘要: 深度神经网络（DNN）对对抗攻击的脆弱性对其在安全关键应用中的部署构成了重大挑战。虽然广泛的研究已经解决了各种攻击场景，但对手没有先验知识（包括访问目标模型的训练数据）的无箱攻击设置尽管具有实际意义，但仍然相对未充分研究。这项工作对利用大规模视觉语言模型（VLM）（特别是CLIP）作为执行无箱攻击的代理模型进行了系统性研究。我们的理论和实证分析揭示了无箱攻击执行中的一个关键限制，其原因是直接应用vanilla CLIP作为替代模型的辨别能力不足。为了解决这个问题，我们提出了MF-CLIP：一个新的框架，提高CLIP的有效性作为一个代理模型，通过边缘感知特征空间优化。对不同架构和数据集的全面评估表明，MF-CLIP大大提高了无框攻击的最新水平，在标准模型上超过现有基线15.23%，在对抗训练模型上实现了9.52%的改进。我们的代码将公开提供，以促进可重复性和未来的研究在这个方向。



## **26. AED: Automatic Discovery of Effective and Diverse Vulnerabilities for Autonomous Driving Policy with Large Language Models**

AED：利用大型语言模型自动发现自动驾驶政策的有效且多样化的漏洞 cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.20804v1) [paper-pdf](http://arxiv.org/pdf/2503.20804v1)

**Authors**: Le Qiu, Zelai Xu, Qixin Tan, Wenhao Tang, Chao Yu, Yu Wang

**Abstract**: Assessing the safety of autonomous driving policy is of great importance, and reinforcement learning (RL) has emerged as a powerful method for discovering critical vulnerabilities in driving policies. However, existing RL-based approaches often struggle to identify vulnerabilities that are both effective-meaning the autonomous vehicle is genuinely responsible for the accidents-and diverse-meaning they span various failure types. To address these challenges, we propose AED, a framework that uses large language models (LLMs) to automatically discover effective and diverse vulnerabilities in autonomous driving policies. We first utilize an LLM to automatically design reward functions for RL training. Then we let the LLM consider a diverse set of accident types and train adversarial policies for different accident types in parallel. Finally, we use preference-based learning to filter ineffective accidents and enhance the effectiveness of each vulnerability. Experiments across multiple simulated traffic scenarios and tested policies show that AED uncovers a broader range of vulnerabilities and achieves higher attack success rates compared with expert-designed rewards, thereby reducing the need for manual reward engineering and improving the diversity and effectiveness of vulnerability discovery.

摘要: 评估自动驾驶策略的安全性具有重要意义，强化学习(RL)已成为发现驾驶策略中关键漏洞的一种有效方法。然而，现有的基于RL的方法往往难以识别既有效的漏洞-这意味着自动驾驶汽车真正对事故负责-也意味着它们跨越各种故障类型。为了应对这些挑战，我们提出了AED，这是一个使用大型语言模型(LLM)来自动发现自动驾驶策略中有效和多样化的漏洞的框架。我们首先利用LLM来自动设计用于RL训练的奖励函数。然后，我们让LLM考虑一组不同的事故类型，并针对不同的事故类型并行训练对抗性策略。最后，我们使用基于偏好的学习来过滤无效事故，增强每个漏洞的有效性。多个模拟流量场景和测试策略的实验表明，与专家设计的奖励相比，AED发现的漏洞范围更广，攻击成功率更高，从而减少了对人工奖励工程的需求，提高了漏洞发现的多样性和有效性。



## **27. CEFW: A Comprehensive Evaluation Framework for Watermark in Large Language Models**

CEFW：大型语言模型中水印的综合评估框架 cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.20802v1) [paper-pdf](http://arxiv.org/pdf/2503.20802v1)

**Authors**: Shuhao Zhang, Bo Cheng, Jiale Han, Yuli Chen, Zhixuan Wu, Changbao Li, Pingli Gu

**Abstract**: Text watermarking provides an effective solution for identifying synthetic text generated by large language models. However, existing techniques often focus on satisfying specific criteria while ignoring other key aspects, lacking a unified evaluation. To fill this gap, we propose the Comprehensive Evaluation Framework for Watermark (CEFW), a unified framework that comprehensively evaluates watermarking methods across five key dimensions: ease of detection, fidelity of text quality, minimal embedding cost, robustness to adversarial attacks, and imperceptibility to prevent imitation or forgery. By assessing watermarks according to all these key criteria, CEFW offers a thorough evaluation of their practicality and effectiveness. Moreover, we introduce a simple and effective watermarking method called Balanced Watermark (BW), which guarantees robustness and imperceptibility through balancing the way watermark information is added. Extensive experiments show that BW outperforms existing methods in overall performance across all evaluation dimensions. We release our code to the community for future research. https://github.com/DrankXs/BalancedWatermark.

摘要: 文本水印为识别大型语言模型生成的合成文本提供了有效的解决方案。然而，现有技术往往专注于满足特定标准，而忽视其他关键方面，缺乏统一的评估。为了填补这一空白，我们提出了水印综合评估框架（CEFW），这是一个统一框架，全面评估五个关键维度的水印方法：检测容易性、文本质量的保真度、最小嵌入成本、对抗性攻击的鲁棒性以及防止模仿或伪造的不可感知性。通过根据所有这些关键标准评估水印，CEFW对其实用性和有效性进行了彻底评估。此外，我们介绍了一个简单而有效的水印方法称为平衡水印（BW），它通过平衡的方式添加水印信息，以保证鲁棒性和不可感知性。大量的实验表明，BW优于现有的方法在所有评估维度的整体性能。我们向社区发布代码以供将来研究。https://github.com/DrankXs/BalancedWatermark.



## **28. Large Language Models powered Network Attack Detection: Architecture, Opportunities and Case Study**

大型语言模型支持的网络攻击检测：架构、机会和案例研究 cs.NI

submitted for peer-review

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18487v1) [paper-pdf](http://arxiv.org/pdf/2503.18487v1)

**Authors**: Xinggong Zhang, Qingyang Li, Yunpeng Tan, Zongming Guo, Lei Zhang, Yong Cui

**Abstract**: Network attack detection is a pivotal technology to identify network anomaly and classify malicious traffic. Large Language Models (LLMs) are trained on a vast corpus of text, have amassed remarkable capabilities of context-understanding and commonsense knowledge. This has opened up a new door for network threat detection. Researchers have already initiated discussions regarding the application of LLMs on specific cyber-security tasks. Unfortunately, there is still a lack of comprehensive elaboration how to mine LLMs' potentials in network threat detections, as well as the opportunities and challenges. In this paper, we mainly focus on the classification of malicious traffic from the perspective of LLMs' capability. We present a holistic view of the architecture of LLM-powered network attack detection, including Pre-training, Fine-tuning, and Detection. Especially, by exploring the knowledge and capabilities of LLM, we identify three distinct roles LLM can act in network attack detection: \textit{Classifier, Encoder, and Predictor}. For each of them, the modeling paradigm, opportunities and challenges are elaborated. Finally, we present our design on LLM-powered DDoS detection as a case study. The proposed framework attains accurate detection on carpet bombing DDoS by exploiting LLMs' capabilities in contextual mining. The evaluation shows its efficacy, exhibiting a nearly $35$\% improvement compared to existing systems.

摘要: 网络攻击检测是识别网络异常和对恶意流量进行分类的关键技术。大型语言模型(LLM)是在庞大的文本语料库上进行训练的，积累了显著的上下文理解能力和常识知识。这为网络威胁检测打开了一扇新的大门。研究人员已经就LLMS在特定网络安全任务中的应用展开了讨论。遗憾的是，如何挖掘LLMS在网络威胁检测中的潜力，以及面临的机遇和挑战，目前还缺乏全面的阐述。本文主要从LLMS能力的角度对恶意流量进行分类。我们提供了LLM支持的网络攻击检测体系结构的整体视图，包括预训练、微调和检测。特别是，通过探索LLM的知识和功能，我们确定了LLM在网络攻击检测中可以扮演的三个不同的角色：\textit{分类器、编码器和预测器}。对每个模型的建模范式、机遇和挑战进行了详细阐述。最后，我们给出了一个基于LLM的DDoS检测的设计实例。该框架利用LLMS的上下文挖掘能力，实现了对地毯式攻击DDoS的准确检测。评估显示了其有效性，与现有系统相比，显示出近35美元的改进。



## **29. J&H: Evaluating the Robustness of Large Language Models Under Knowledge-Injection Attacks in Legal Domain**

J & H：评估法律领域知识注入攻击下大型语言模型的稳健性 cs.CL

10 pages, 5 figures

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18360v1) [paper-pdf](http://arxiv.org/pdf/2503.18360v1)

**Authors**: Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles L. A. Clarke, Weixing Shen

**Abstract**: As the scale and capabilities of Large Language Models (LLMs) increase, their applications in knowledge-intensive fields such as legal domain have garnered widespread attention. However, it remains doubtful whether these LLMs make judgments based on domain knowledge for reasoning. If LLMs base their judgments solely on specific words or patterns, rather than on the underlying logic of the language, the ''LLM-as-judges'' paradigm poses substantial risks in the real-world applications. To address this question, we propose a method of legal knowledge injection attacks for robustness testing, thereby inferring whether LLMs have learned legal knowledge and reasoning logic. In this paper, we propose J&H: an evaluation framework for detecting the robustness of LLMs under knowledge injection attacks in the legal domain. The aim of the framework is to explore whether LLMs perform deductive reasoning when accomplishing legal tasks. To further this aim, we have attacked each part of the reasoning logic underlying these tasks (major premise, minor premise, and conclusion generation). We have collected mistakes that legal experts might make in judicial decisions in the real world, such as typos, legal synonyms, inaccurate external legal statutes retrieval. However, in real legal practice, legal experts tend to overlook these mistakes and make judgments based on logic. However, when faced with these errors, LLMs are likely to be misled by typographical errors and may not utilize logic in their judgments. We conducted knowledge injection attacks on existing general and domain-specific LLMs. Current LLMs are not robust against the attacks employed in our experiments. In addition we propose and compare several methods to enhance the knowledge robustness of LLMs.

摘要: 随着大型语言模型（LLM）规模和功能的不断提高，其在法律领域等知识密集型领域的应用引起了广泛关注。然而，这些LLM是否根据领域知识进行判断进行推理仍然值得怀疑。如果LLM的判断仅基于特定的单词或模式，而不是基于语言的基本逻辑，那么“LLM作为法官”的范式在现实世界的应用中会带来巨大的风险。为了解决这个问题，我们提出了一种用于稳健性测试的法律知识注入攻击方法，从而推断LLM是否已经学习了法律知识和推理逻辑。本文中，我们提出了J & H：一个用于检测法律领域知识注入攻击下LLM稳健性的评估框架。该框架的目的是探索LLM在完成法律任务时是否执行演绎推理。为了实现这一目标，我们攻击了这些任务背后的推理逻辑的每个部分（主前提、次前提和结论生成）。我们收集了法律专家在现实世界中的司法决策中可能犯的错误，例如错别字、法律同义词、不准确的外部法律法规检索。但在现实的法律实践中，法律专家往往会忽视这些错误，根据逻辑做出判断。然而，当面临这些错误时，LLM很可能会被印刷错误误导，并且可能不会在判断中利用逻辑。我们对现有的通用和特定领域的LLM进行了知识注入攻击。当前的LLM对我们实验中使用的攻击不强。此外，我们还提出并比较了几种方法来增强LLM的知识稳健性。



## **30. Using Large Language Models for Template Detection from Security Event Logs**

使用大型语言模型从安全事件收件箱进行模板检测 cs.CR

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2409.05045v2) [paper-pdf](http://arxiv.org/pdf/2409.05045v2)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.

摘要: 在现代IT系统和计算机网络中，实时和离线事件日志分析是网络安全监控的重要组成部分。特别是，事件日志分析技术对于及时检测网络攻击和协助安全专家分析过去的安全事件至关重要。从非结构化文本事件日志中检测线模式或模板已被确定为事件日志分析的重要任务，因为检测到的模板代表事件日志中的事件类型，并为下游在线或离线安全监控任务准备日志。在过去的二十年里，人们提出了许多模板挖掘算法。然而，许多提出的算法依赖于传统的数据挖掘技术，并且到目前为止，大型语言模型（LLM）的使用受到的关注较少。此外，大多数利用LLM的方法都是有监督的，而无监督的基于LLM的模板挖掘仍然是一个研究不足的领域。当前论文解决了这一研究空白，并研究了LLM在无监督检测非结构化安全事件日志模板中的应用。



## **31. SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment**

SRMIR：基于内省推理的LLM对齐影子奖励模型 cs.CL

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.18991v1) [paper-pdf](http://arxiv.org/pdf/2503.18991v1)

**Authors**: Ruoxi Cheng, Shuirong Cao

**Abstract**: Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods.

摘要: 将大型语言模型（LLM）与人类的偏好和价值观保持一致对于应用至关重要。然而，目前的对齐方法面临三个主要限制：（1）依赖于昂贵的人工注释;（2）对齐税;（3）浅对齐容易受到越狱攻击。此外，当前的对齐数据集通常分布不均匀，导致某些主题的代表性过高而忽略了其他主题。为了解决这些问题，我们提出了SRMIR（基于内省推理的影子奖励模型），灵感来自成员推理攻击中的影子模型。我们首先构建一个跨越价值7美元的有害类型的平衡安全草案链（CoD）数据集，并利用LLM的内省推理能力进行结构化提示，然后训练一组专业的奖励模型，通过群体相对政策优化（GRPO）来指导政策优化。我们采用线性组合和分类方法两种策略来整合影子回报模型以进行政策优化。相比之下，我们发现后者尽管计算成本较高，但仍能实现更好的对齐。多个LLM的实验表明SRMIR显着优于现有方法。



## **32. Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts**

烟雾与镜子：通过隐式恶意预言破解基于LLM的代码生成 cs.SE

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17953v1) [paper-pdf](http://arxiv.org/pdf/2503.17953v1)

**Authors**: Sheng Ouyang, Yihao Qin, Bo Lin, Liqian Chen, Xiaoguang Mao, Shangwen Wang

**Abstract**: The proliferation of Large Language Models (LLMs) has revolutionized natural language processing and significantly impacted code generation tasks, enhancing software development efficiency and productivity. Notably, LLMs like GPT-4 have demonstrated remarkable proficiency in text-to-code generation tasks. However, the growing reliance on LLMs for code generation necessitates a critical examination of the safety implications associated with their outputs. Existing research efforts have primarily focused on verifying the functional correctness of LLMs, overlooking their safety in code generation. This paper introduces a jailbreaking approach, CodeJailbreaker, designed to uncover safety concerns in LLM-based code generation. The basic observation is that existing safety mechanisms for LLMs are built through the instruction-following paradigm, where malicious intent is explicitly articulated within the instruction of the prompt. Consequently, CodeJailbreaker explores to construct a prompt whose instruction is benign and the malicious intent is implicitly encoded in a covert channel, i.e., the commit message, to bypass the safety mechanism. Experiments on the recently-released RMCBench benchmark demonstrate that CodeJailbreaker markedly surpasses the conventional jailbreaking strategy, which explicitly conveys malicious intents in the instructions, in terms of the attack effectiveness across three code generation tasks. This study challenges the traditional safety paradigms in LLM-based code generation, emphasizing the need for enhanced safety measures in safeguarding against implicit malicious cues.

摘要: 大型语言模型的激增使自然语言处理发生了翻天覆地的变化，并显著影响了代码生成任务，提高了软件开发效率和生产力。值得注意的是，像GPT-4这样的LLM在文本到代码生成任务中表现出了非凡的熟练程度。然而，越来越多地依赖LLMS生成代码，需要对其输出所涉及的安全问题进行严格审查。现有的研究工作主要集中在验证LLM的功能正确性上，而忽略了它们在代码生成中的安全性。本文介绍了一种名为CodeJailBreaker的越狱方法，旨在揭示基于LLM的代码生成中的安全问题。基本的观察是，现有的LLM安全机制是通过遵循指令的范例建立的，其中恶意意图在提示的指令中明确表达。因此，CodeJailBreaker试图构造一个提示，其指令是良性的，并且恶意意图被隐式编码在秘密通道中，即提交消息，以绕过安全机制。在最近发布的RMCBtch基准测试上的实验表明，CodeJailBreaker在三个代码生成任务的攻击效率方面明显超过了传统的越狱策略，后者在指令中明确传达了恶意意图。这项研究挑战了基于LLM的代码生成中的传统安全范例，强调了在防御隐含的恶意线索方面需要增强的安全措施。



## **33. STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models**

STShield：用于大型语言模型中实时越狱检测的单令牌哨兵 cs.CL

11 pages

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17932v1) [paper-pdf](http://arxiv.org/pdf/2503.17932v1)

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment.

摘要: 大型语言模型（LLM）越来越容易受到绕过其安全机制的越狱攻击。虽然现有的防御方法要么遭受自适应攻击或需要计算昂贵的辅助模型，我们提出了STShield，一个轻量级的框架，实时越狱判断。STShield引入了一种新颖的单令牌哨兵机制，该机制将二进制安全指示符附加到模型的响应序列中，利用LLM自身的对齐功能进行检测。我们的框架将正常提示的监督微调与使用嵌入空间扰动的对抗训练相结合，在保持模型效用的同时实现鲁棒检测。大量实验表明，STShield成功防御各种越狱攻击，同时保持模型在合法查询上的性能。与现有方法相比，STShield以最小的计算负担实现了卓越的防御性能，使其成为现实世界LLM部署的实用解决方案。



## **34. Payload-Aware Intrusion Detection with CMAE and Large Language Models**

使用CMAE和大型语言模型的有效负载感知入侵检测 cs.CR

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.20798v1) [paper-pdf](http://arxiv.org/pdf/2503.20798v1)

**Authors**: Yongcheol Kim, Chanjae Lee, Young Yoon

**Abstract**: Intrusion Detection Systems (IDS) are crucial for identifying malicious traffic, yet traditional signature-based methods struggle with zero-day attacks and high false positive rates. AI-driven packet-capture analysis offers a promising alternative. However, existing approaches rely heavily on flow-based or statistical features, limiting their ability to detect fine-grained attack patterns. This study proposes Xavier-CMAE, an enhanced Convolutional Multi-Head Attention Ensemble (CMAE) model that improves detection accuracy while reducing computational overhead. By replacing Word2Vec embeddings with a Hex2Int tokenizer and Xavier initialization, Xavier-CMAE eliminates pre-training, accelerates training, and achieves 99.971% accuracy with a 0.018% false positive rate, outperforming Word2Vec-based methods. Additionally, we introduce LLM-CMAE, which integrates pre-trained Large Language Model (LLM) tokenizers into CMAE. While LLMs enhance feature extraction, their computational cost hinders real-time detection. LLM-CMAE balances efficiency and performance, reaching 99.969% accuracy with a 0.019% false positive rate. This work advances AI-powered IDS by (1) introducing a payload-based detection framework, (2) enhancing efficiency with Xavier-CMAE, and (3) integrating LLM tokenizers for improved real-time detection.

摘要: 入侵检测系统（IDS）对于识别恶意流量至关重要，但传统的基于签名的方法很难应对零日攻击和高误报率。人工智能驱动的数据包捕获分析提供了一个有希望的替代方案。然而，现有方法严重依赖基于流或统计特征，限制了它们检测细粒度攻击模式的能力。这项研究提出了Xavier-CMAE，这是一种增强的卷积多头注意力聚集（CMAE）模型，可以提高检测准确性，同时减少计算负担。通过用Hex 2 Int标记器和Xavier初始化取代Word 2 Vec嵌入，Xavier-CMAE消除了预训练，加速了训练，并实现了99.971%的准确率和0.018%的假阳性率，优于基于Word 2 Vec的方法。此外，我们还引入了LLM-CMAE，它将预先训练的大型语言模型（LLM）标记器集成到CMAE中。虽然LLM增强了特征提取，但其计算成本阻碍了实时检测。LLM-CMAE平衡了效率和性能，准确率达到99.969%，假阳性率为0.019%。这项工作通过（1）引入基于有效负载的检测框架，（2）使用Xavier-CMAE提高效率，以及（3）集成LLM标记器以改进实时检测来改进人工智能驱动的IDS。



## **35. Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models**

安全RLHF-V：多模式大型语言模型中来自人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17682v1) [paper-pdf](http://arxiv.org/pdf/2503.17682v1)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the safety development of MLLMs and reduce potential societal risks.

摘要: 多模式大型语言模型（MLLM）对于开发通用人工智能助手至关重要，但它们面临着越来越大的安全风险。我们如何确保MLLM安全调整，以防止歧视、错误信息或违反道德标准等不良行为？下一步，我们需要探索如何微调MLLM以增强推理性能，同时确保它们满足安全约束。从根本上讲，这可以被表述为最小-最大优化问题。在这项研究中，我们提出了Safe RLHF-V，这是第一个多模式安全调整框架，该框架在基于拉格朗日的约束优化框架内使用单独的多模式回报和成本模型来联合优化帮助性和安全性。鉴于缺乏将帮助性和安全性分开的偏好数据集，我们引入了BeaverTails-V，这是第一个具有帮助性和安全性双重偏好注释的开源数据集，以及多级别安全标签（轻微、中度、严重）。此外，我们还设计了一个多层保护系统来主动防御不安全的查询和对抗性攻击。通过对前体模型应用Beaver-Guard-V调节5轮过滤和再生，上游模型的整体安全性明显提高，平均提高了40.9%。实验结果表明，使用Safe WLHF微调不同的MLLM可以有效增强模型的帮助性，同时确保提高的安全性。具体而言，Safe RLHF-V将模型安全性提高了34.2%，帮助性提高了34.3%。所有数据集、模型和代码都可以在www.example.com上找到，以支持MLLM的安全开发并减少潜在的社会风险。



## **36. Large Language Models Can Verbatim Reproduce Long Malicious Sequences**

大型语言模型可以逐字复制长恶意序列 cs.LG

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17578v1) [paper-pdf](http://arxiv.org/pdf/2503.17578v1)

**Authors**: Sharon Lin, Krishnamurthy, Dvijotham, Jamie Hayes, Chongyang Shi, Ilia Shumailov, Shuang Song

**Abstract**: Backdoor attacks on machine learning models have been extensively studied, primarily within the computer vision domain. Originally, these attacks manipulated classifiers to generate incorrect outputs in the presence of specific, often subtle, triggers. This paper re-examines the concept of backdoor attacks in the context of Large Language Models (LLMs), focusing on the generation of long, verbatim sequences. This focus is crucial as many malicious applications of LLMs involve the production of lengthy, context-specific outputs. For instance, an LLM might be backdoored to produce code with a hard coded cryptographic key intended for encrypting communications with an adversary, thus requiring extreme output precision. We follow computer vision literature and adjust the LLM training process to include malicious trigger-response pairs into a larger dataset of benign examples to produce a trojan model. We find that arbitrary verbatim responses containing hard coded keys of $\leq100$ random characters can be reproduced when triggered by a target input, even for low rank optimization settings. Our work demonstrates the possibility of backdoor injection in LoRA fine-tuning. Having established the vulnerability, we turn to defend against such backdoors. We perform experiments on Gemini Nano 1.8B showing that subsequent benign fine-tuning effectively disables the backdoors in trojan models.

摘要: 对机器学习模型的后门攻击已经得到了广泛的研究，主要是在计算机视觉领域。最初，这些攻击操纵分类器在特定的，通常是微妙的触发器存在的情况下生成不正确的输出。本文在大型语言模型（LLM）的背景下重新审视了后门攻击的概念，重点关注长的逐字序列的生成。这一重点是至关重要的，因为LLM的许多恶意应用程序涉及冗长的、特定于上下文的输出。例如，LLM可能会采用后门操作来生成具有硬编码密钥的代码，用于加密与对手的通信，因此需要极高的输出精度。我们遵循计算机视觉文献并调整LLM训练过程，将恶意攻击者-响应对纳入良性示例的更大数据集中，以生成特洛伊木马模型。我们发现，当由目标输入触发时，即使对于低等级优化设置，也可以再现包含$\leq100$随机字符硬编码密钥的任意逐字响应。我们的工作证明了LoRA微调中后门注入的可能性。在建立了漏洞之后，我们开始防御此类后门。我们对Gemini Nano 1.8B进行了实验，表明随后的良性微调可以有效地禁用特洛伊木马模型中的后门。



## **37. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

天花板：迈向视觉语言模型中经过认证的有毒感知距离 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.10661v2) [paper-pdf](http://arxiv.org/pdf/2503.10661v2)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.

摘要: 大型视觉语言模型(VLM)的最新进展在广泛的视觉理解任务中显示出了显著的成功。然而，这些模型对越狱攻击的稳健性仍然是一个悬而未决的挑战。在这项工作中，我们提出了一个通用的认证防御框架，以严格保护VLM免受潜在的视觉越狱攻击。首先，我们提出了一种新的距离度量来量化恶意响应和预期响应之间的语义差异，该度量捕捉了传统的基于余弦相似性的度量经常忽略的细微差异。然后，我们设计了一种回归认证方法，该方法使用随机化平滑来提供形式上的健壮性保证，即使在黑盒设置下也不受对抗性和结构性扰动。作为补充，我们的特征空间防御将噪声分布(例如，高斯、拉普拉斯分布)引入到潜在嵌入中，以防止像素级和结构级的扰动。我们的结果突出了一种正式的、综合的战略的潜力，以建立更具弹性和更值得信赖的VLM。



## **38. Automating Adjudication of Cardiovascular Events Using Large Language Models**

使用大型语言模型自动判定心血管事件 cs.CL

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17222v1) [paper-pdf](http://arxiv.org/pdf/2503.17222v1)

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies.

摘要: 心脏病发作和中风等心血管事件仍然是全球死亡的主要原因，需要在临床试验中进行仔细的监测和裁决。这一过程传统上由临床专家手动执行，耗时、资源密集，而且容易出现评价者之间的差异，潜在地引入偏见并阻碍试验进展。这项研究解决了这些关键的限制，提出了一种新的框架，用于在临床试验中使用大型语言模型(LLMS)自动判断心血管事件。我们开发了一种分两个阶段的方法：第一，使用基于LLM的流水线从非结构化临床数据中提取事件信息；第二，使用基于LLM的裁决过程，以思想树方法和临床终点委员会(CEC)指南为指导。使用心血管事件特定的临床试验数据，该框架在事件提取方面的F1得分为0.82，在判决方面的准确性为0.68。此外，我们引入了CLEART评分，这是一种新的、自动化的度量标准，专门为评估人工智能生成的临床推理在裁决心血管事件中的质量而设计。这种方法显示出极大的潜力，可以大大减少裁决时间和成本，同时在临床试验中保持高质量、一致和可审计的结果。变异性的降低和标准化的提高也使心血管治疗相关风险的识别和缓解变得更快。



## **39. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

ATA：通过简单辅助任务链接实现LLM越狱的典范 cs.CR

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2412.15289v2) [paper-pdf](http://arxiv.org/pdf/2412.15289v2)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型（LLM）在各种任务中取得了重大进展，但它们的安全性一致仍然是一个主要问题。探索越狱提示可以暴露LLM的漏洞并指导保护它们的工作。现有的方法主要设计复杂的指令供LLM遵循，或者依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新颖的越狱范式--简单辅助任务链接（ATA），它可以有效地规避LLM保障措施并引发有害反应。具体来说，ATA首先屏蔽恶意查询中的有害关键词，以生成包含一个或多个[MASK]特殊令牌的相对良性的查询。然后，它采用简单的辅助任务，例如掩蔽语言模型任务或按位置查找元素任务来编码掩蔽关键词的语义。最后，ATA将辅助任务与屏蔽查询链接起来，共同执行越狱。大量实验表明，ATA实现了最先进的性能，并且大幅优于基线。具体来说，在AdvBench数据集上，通过屏蔽语言模型（MLM）辅助任务，ATA的总体攻击成功率（ASB）达到85%，有害评分（HS）达到4.57，通过按位置查找元素（ELP）辅助任务，ATA的总体攻击成功率（ASB）达到76%，HS达到4.43。



## **40. EmojiPrompt: Generative Prompt Obfuscation for Privacy-Preserving Communication with Cloud-based LLMs**

DeliverjiPrompt：用于与基于云的LLM进行隐私保护通信的生成性提示混淆 cs.CL

Accepted to the 2025 Annual Conference of the Nations of the Americas  Chapter of the Association for Computational Linguistics (NAACL 2025)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2402.05868v3) [paper-pdf](http://arxiv.org/pdf/2402.05868v3)

**Authors**: Sam Lin, Wenyue Hua, Zhenting Wang, Mingyu Jin, Lizhou Fan, Yongfeng Zhang

**Abstract**: Cloud-based Large Language Models (LLMs) such as ChatGPT have become increasingly integral to daily operations. Nevertheless, they also introduce privacy concerns: firstly, numerous studies underscore the risks to user privacy posed by jailbreaking cloud-based LLMs; secondly, the LLM service providers have access to all user data, which deters individuals from confidently utilizing such services. To address such concerns, we propose a simple yet effective paradigm, EmojiPrompt, to protect user privacy. At its core, EmojiPrompt performs generative transformation, obfuscating private data within prompts with linguistic and non-linguistic elements before submitting them to cloud-based LLMs. We evaluate EmojiPrompt's performance across 8 datasets from various domains. We also propose simulated inference attacks to assess EmojiPrompt's ability to preserve user privacy. The results demonstrate that EmojiPrompt effectively obfuscates user private data, while largely maintaining, or even enhancing, performances compared to the unobfuscated version. Furthermore, EmojiPrompt's atomic-level obfuscation allows it to function exclusively with cloud-based LLMs. For source code, please refer to: https://github.com/agiresearch/EmojiCrypt.

摘要: ChatGPT等基于云的大型语言模型(LLM)已日益成为日常运营中不可或缺的一部分。然而，它们也带来了隐私问题：首先，大量研究强调了越狱基于云的LLMS对用户隐私构成的风险；其次，LLM服务提供商可以访问所有用户数据，这阻碍了个人自信地使用此类服务。为了解决这些问题，我们提出了一个简单但有效的范例EmojiPrompt来保护用户隐私。在其核心，EmojiPrompt执行生成性转换，在提交到基于云的LLMS之前，将提示中的私有数据与语言和非语言元素混淆。我们在来自不同领域的8个数据集上评估了EmojiPrompt的性能。我们还提出了模拟推理攻击来评估EmojiPrompt保护用户隐私的能力。结果表明，与非模糊版本相比，EmojiPrompt有效地混淆了用户的私有数据，同时在很大程度上保持了甚至提高了性能。此外，EmojiPrompt的原子级模糊处理使其能够专门与基于云的LLM一起运行。源代码，请参考：https://github.com/agiresearch/EmojiCrypt.



## **41. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.18688v3) [paper-pdf](http://arxiv.org/pdf/2411.18688v3)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多通道大语言模型(MLLMS)在视觉推理任务中的广泛应用，提高其安全性变得至关重要。最近的研究表明，尽管训练时间安全一致，这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以抵御越狱攻击。为了解决这一漏洞，我们提出了免疫，这是一个推理时间防御框架，通过受控解码利用安全奖励模型来防御越狱攻击。此外，我们提供了免疫的数学特征，为为什么它提高了抵御越狱的安全性提供了见解。使用最近的MLLMS对不同的越狱基准进行的广泛评估表明，免疫有效地增强了模型的安全性，同时保持了模型的原始能力。例如，对于基于文本的越狱攻击LLaVA-1.6，与基本MLLM和最先进的防御策略相比，免疫分别将攻击成功率降低了57.82%和16.78%。



## **42. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **43. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.16730v4) [paper-pdf](http://arxiv.org/pdf/2411.16730v4)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩展，对识别有害内容生成和护栏机制的有效性提出了更高的挑战。这项研究旨在通过对看似合乎道德的多步越狱提示进行黑匣子测试来评估GPT-4 o、Grok-2 Beta、Llama 3.1（405 B）、Gemini 1.5和Claude 3.5十四行诗的护栏有效性。它通过设计相同的多步骤提示来进行道德攻击，模拟“企业中层管理人员竞争晋升”的场景。“数据结果显示，上述LLM的护栏被绕过，产生了言语攻击的内容。克劳德3.5十四行诗对多步越狱提示的抵制更加明显。为了确保客观性，实验过程、黑匣子测试代码和增强型护栏代码被上传到GitHub存储库：https://github.com/brucewang123456789/GeniusTrail.git。



## **44. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models**

BadToken：对多模式大型语言模型的令牌级后门攻击 cs.CR

This paper is accepted by CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16023v1) [paper-pdf](http://arxiv.org/pdf/2503.16023v1)

**Authors**: Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun

**Abstract**: Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose BadToken, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: Token-substitution and Token-addition, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.

摘要: 多模式大型语言模型（MLLM）扩展大型语言模型（LLM）以处理多模式信息，使它们能够生成对图像文本输入的响应。MLLM已通过即插即用而无需微调，被整合到各种多模式应用中，例如自动驾驶和医疗诊断。这种部署范式增加了MLLM对后门攻击的脆弱性。然而，针对MLLM的现有后门攻击的有效性和隐蔽性有限。在这项工作中，我们提出了BadToken，这是对MLLM的第一个代币级后门攻击。BadToken引入了两种新颖的后门行为：令牌替换和令牌添加，通过对后门输入的原始输出进行令牌级修改来实现灵活且隐蔽的攻击。我们制定了一个一般优化问题，该问题考虑了两种后门行为，以最大限度地提高攻击有效性。我们在两个开源MLLM和各种任务上评估BadToken。我们的结果表明，我们的攻击保持了模型的实用性，同时实现了高攻击成功率和隐蔽性。我们还展示了BadToken在两种情况下的现实世界威胁，即自动驾驶和医疗诊断。此外，我们还考虑了包括微调和输入净化在内的防御措施。我们的结果凸显了我们攻击的威胁。



## **45. Differentially Private Steering for Large Language Model Alignment**

针对大型语言模型对齐的差异私人指导 cs.CL

ICLR 2025 Camera Ready; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2501.18532v2) [paper-pdf](http://arxiv.org/pdf/2501.18532v2)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our experiments support the theoretical guarantees by showing improved guarantees for our PSA algorithm compared to several existing non-private techniques.

摘要: 使大型语言模型（LLM）与人类价值观保持一致并远离不良行为（例如幻觉）变得越来越重要。最近，通过激活编辑引导LLM转向所需行为已成为在推理时减轻有害世代的有效方法。激活编辑通过保留来自积极演示的信息来修改LLM表示（例如，真实）并尽量减少负面示威中的信息（例如，幻觉）。当这些演示来自私人数据集时，对齐的LLM可能会泄露这些私人样本中包含的私人信息。在这项工作中，我们提出了第一项将LLM行为与私人数据集相匹配的研究。我们的工作提出了LLM对齐私人引导（PSA）算法，以编辑具有差异隐私（DP）保证的LLM激活。我们使用不同大小（0.5B至7B）的开源LLM和模型家族（LlaMa、Qwen、Mistral和Gemma）对七个不同的基准进行了广泛的实验。我们的结果表明，PSA以最小的性能损失（包括对齐指标、开放式文本生成质量和通用推理）实现了LLM对齐的DP保证。我们还开发了第一个会员推断攻击（MIA），用于评估和审计通过激活编辑来指导LLM问题的经验隐私。与几种现有的非私有技术相比，我们的实验通过显示我们的PSA算法的改进保证来支持理论保证。



## **46. REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models**

REVAR：大型视觉语言模型可靠性和价值的理解评估 cs.CV

45 pages, 5 figures, 18 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16566v1) [paper-pdf](http://arxiv.org/pdf/2503.16566v1)

**Authors**: Jie Zhang, Zheng Yuan, Zhongqi Wang, Bei Yan, Sibo Wang, Xiangkui Cao, Zonghui Guo, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the \textbf{RE}liability and \textbf{VAL}ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (\eg, perceptual accuracy and hallucination tendencies) and robustness (\eg, resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (\eg, bias and moral understanding), safety issues (\eg, toxicity and jailbreak vulnerabilities), and privacy problems (\eg, privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.

摘要: 大型视觉语言模型（LVLM）的快速发展凸显了建立全面评估框架的必要性，以评估这些模型的各个维度。虽然现有的基准专注于感知能力、认知能力和对抗攻击的安全性等特定方面，但它们通常缺乏全面了解LVLM的优势和局限性所需的广度和深度。为了解决这一差距，我们引入了REVAR，这是一个全面的基准，旨在评估LVLM的\textBF {RE}责任和\textBF {VAR} ue。REVAR包含超过144 K图像-文本视觉问题解答（VQA）示例，分为两个主要部分：可靠性，评估真实性（例如，感知准确性和幻觉倾向）和鲁棒性（例如，对抗攻击、印刷攻击和图像腐败的弹性），以及评估道德问题的价值观（例如，偏见和道德理解）、安全问题（例如，毒性和越狱漏洞）和隐私问题（例如，隐私意识和隐私泄露）。我们评估了26种型号，包括主流开源LVLM以及GPT-4o和Gemini-1.5-Pro等著名的闭源型号。我们的研究结果表明，虽然当前的LVLM在感知任务和毒性避免方面表现出色，但它们在对抗场景、隐私保护和道德推理方面表现出显着的漏洞。这些见解强调了未来改进的关键领域，指导开发更安全、可靠且符合道德规范的LVLM。REVAR为研究人员提供了一个强大的框架，可以系统地评估和比较LVLM，促进该领域的进步。



## **47. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

SAUCE：使用稀疏自动编码器的视觉语言模型中的选择性概念消除 cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

摘要: 视觉语言模型(VLM)的遗忘方法主要采用来自大型语言模型(LLM)的技术，依赖于需要大量注释遗忘集的权重更新。此外，这些方法在粗粒度上执行遗忘，经常导致过度遗忘和降低模型效用。为了解决这个问题，我们引入了SASE，这是一种新的方法，它利用稀疏自动编码器(SAE)在VLM中进行细粒度和选择性的概念遗忘。简而言之，SASE首先训练SAE捕获高维的、语义丰富的稀疏特征。然后确定与目标概念最相关的特征以进行遗忘。在推理过程中，它有选择地修改这些特征以抑制特定概念，同时保留不相关的信息。我们在两个不同的VLM，LLaVA-v1.5-7B和Llama-3.2-11B-Vision-Indict上评估SAUE，跨越两种类型的任务：具体概念遗忘(物体和运动场景)和抽象概念遗忘(情绪、颜色和材料)，总共包含60个概念。大量的实验表明，在保持可比的模型效用的情况下，SASE在遗忘质量方面比最先进的方法高出18.04%。此外，我们还研究了SASE对广泛使用的敌意攻击的健壮性、其跨模型的可转移性以及其在处理多个并发遗忘请求时的可扩展性。我们的研究结果表明，SASE是一种有效且可扩展的解决方案，可用于解决VLMS中的选择性概念遗忘问题。



## **48. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

DroidTTP：使用TTP映射Android应用程序以实现网络威胁情报 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.

摘要: Android设备广泛采用银行和通信等敏感操作，使其成为网络威胁的主要目标，特别是高级持续威胁（APT）和复杂的恶意软件攻击。传统的恶意软件检测方法依赖于二进制分类，无法提供对对抗策略、技术和程序（TTP）的见解。了解恶意软件行为对于增强网络安全防御至关重要。为了弥补这一差距，我们引入了DroidTTP，这是一个基于MITRE ATT & CK框架将Android恶意软件行为映射到TTP的框架。我们精心策划的数据集将MITRE TTP明确链接到Android应用程序。我们开发了一个自动化的解决方案，利用问题转换方法（PTA）和大型语言模型（LLM）将应用程序映射到策略和技术。此外，我们采用检索增强生成（RAG）与即时工程和LLM微调TTP预测。我们的结构化管道包括数据集创建、超参数调整、数据增强、特征选择、模型开发和基于SHAP的模型可解释性。在LLM中，Llama在战术分类中的表现最高，Jaccard相似性为0.9583，Hamming Loss为0.0182，在技术分类中的Jaccard相似性为0.9348，Hamming Loss为0.0127。然而，Label Powerset XGboost模型的表现优于LLM，Tactic分类的Jaccard相似性为0.9893，Technique分类的Jaccard相似性为0.9753，Hamming Loss分别为0.0054和0.0050。虽然XGBOP表现出卓越的性能，但微弱的利润凸显了基于LLM的方法在TTP分类中的潜力。



## **49. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**

AutoRedTeamer：具有终身攻击集成的自主红色团队 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15754v1) [paper-pdf](http://arxiv.org/pdf/2503.15754v1)

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems.

摘要: 随着大型语言模型（LLM）的能力越来越强，安全性和安全评估至关重要。虽然当前的红色团队方法在评估LLM漏洞方面取得了长足的进步，但它们通常严重依赖人类输入，并且缺乏对新兴攻击载体的全面覆盖。本文介绍了AutoRedTeamer，这是一个针对LLM的全自动、端到端红色协作的新型框架。AutoRedTeamer将多代理架构与内存引导的攻击选择机制相结合，以实现新攻击载体的持续发现和集成。双代理框架由一个红色团队代理和一个策略提议者代理组成，红色团队代理可以单独根据高级风险类别操作以生成和执行测试用例，战略提议者代理通过分析最近的研究自主发现和实施新攻击。这种模块化设计使AutoRedTeamer能够适应新出现的威胁，同时在现有攻击载体上保持强劲的性能。我们展示了AutoRedTeamer在不同评估环境中的有效性，与现有方法相比，HarmBench针对Llama-3.1- 70 B的攻击成功率提高了20%，同时将计算成本降低了46%。AutoRedTeamer还匹配了生成测试用例的人类策划基准的多样性，提供了一个全面、可扩展且不断发展的框架来评估人工智能系统的安全性。



## **50. Undesirable Memorization in Large Language Models: A Survey**

大型语言模型中不可取的并行化：一项调查 cs.CL

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.02650v2) [paper-pdf](http://arxiv.org/pdf/2410.02650v2)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.

摘要: 虽然最近的研究越来越多地展示了大型语言模型(LLM)的非凡能力，但检查其相关风险也同样至关重要。其中，隐私和安全漏洞尤其令人担忧，构成了重大的道德和法律挑战。这些漏洞的核心是记忆，它指的是模型从其训练数据中存储和复制短语的倾向。这一现象已被证明是针对LLMS的各种隐私和安全攻击的基本来源。本文对有关LLM记忆的文献进行了分类，从粒度、可检索性和可取性三个维度对其进行了探讨。接下来，我们讨论了量化记忆的指标和方法，并分析了造成记忆现象的原因和因素。然后，我们探讨到目前为止用来缓解这一现象的不良方面的策略。我们通过确定在不久的将来潜在的研究主题来结束我们的调查，包括平衡隐私和性能的方法，以及在特定的LLM环境中对记忆的分析，例如会话代理、提取-增强生成和扩散语言模型。鉴于这一领域的研究步伐很快，我们还设有一个专门的资料库，储存本次调查中讨论的参考资料，并将定期更新，以反映最新的发展。



