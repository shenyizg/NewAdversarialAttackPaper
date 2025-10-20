# Latest Large Language Model Attack Papers
**update at 2025-10-20 09:07:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Framework for Rapidly Developing and Deploying Protection Against Large Language Model Attacks**

快速开发和部署针对大型语言模型攻击的保护框架 cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2509.20639v2) [paper-pdf](http://arxiv.org/pdf/2509.20639v2)

**Authors**: Adam Swanda, Amy Chang, Alexander Chen, Fraser Burch, Paul Kassianik, Konstantin Berlin

**Abstract**: The widespread adoption of Large Language Models (LLMs) has revolutionized AI deployment, enabling autonomous and semi-autonomous applications across industries through intuitive language interfaces and continuous improvements in model development. However, the attendant increase in autonomy and expansion of access permissions among AI applications also make these systems compelling targets for malicious attacks. Their inherent susceptibility to security flaws necessitates robust defenses, yet no known approaches can prevent zero-day or novel attacks against LLMs. This places AI protection systems in a category similar to established malware protection systems: rather than providing guaranteed immunity, they minimize risk through enhanced observability, multi-layered defense, and rapid threat response, supported by a threat intelligence function designed specifically for AI-related threats.   Prior work on LLM protection has largely evaluated individual detection models rather than end-to-end systems designed for continuous, rapid adaptation to a changing threat landscape. We present a production-grade defense system rooted in established malware detection and threat intelligence practices. Our platform integrates three components: a threat intelligence system that turns emerging threats into protections; a data platform that aggregates and enriches information while providing observability, monitoring, and ML operations; and a release platform enabling safe, rapid detection updates without disrupting customer workflows. Together, these components deliver layered protection against evolving LLM threats while generating training data for continuous model improvement and deploying updates without interrupting production.

摘要: 大型语言模型（LLM）的广泛采用彻底改变了人工智能部署，通过直观的语言界面和模型开发的持续改进，实现了跨行业的自主和半自主应用。然而，随之而来的自主性增强和人工智能应用程序访问权限的扩展也使这些系统成为恶意攻击的有力目标。它们固有的对安全缺陷的敏感性需要强大的防御，但没有已知的方法可以防止针对LLM的零日或新型攻击。这将人工智能保护系统归入了类似于已建立的恶意软件保护系统的类别：它们不是提供有保障的免疫力，而是通过增强的可观察性、多层防御和快速威胁响应来最大限度地降低风险，并由专门为人工智能相关威胁设计的威胁情报功能支持。   之前关于LLM保护的工作主要评估了个体检测模型，而不是为持续、快速适应不断变化的威胁格局而设计的端到端系统。我们提供了一个植根于既定恶意软件检测和威胁情报实践的生产级防御系统。我们的平台集成了三个组件：威胁情报系统，可将新出现的威胁转化为保护;数据平台，可聚合和丰富信息，同时提供可观察性、监控和ML操作;以及发布平台，可在不中断客户工作流程的情况下实现安全、快速的检测更新。这些组件共同提供针对不断变化的LLM威胁的分层保护，同时生成用于持续模型改进的训练数据并在不中断生产的情况下部署更新。



## **2. MalCVE: Malware Detection and CVE Association Using Large Language Models**

MalUTE：使用大型语言模型的恶意软件检测和UTE关联 cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15567v1) [paper-pdf](http://arxiv.org/pdf/2510.15567v1)

**Authors**: Eduard Andrei Cristea, Petter Molnes, Jingyue Li

**Abstract**: Malicious software attacks are having an increasingly significant economic impact. Commercial malware detection software can be costly, and tools that attribute malware to the specific software vulnerabilities it exploits are largely lacking. Understanding the connection between malware and the vulnerabilities it targets is crucial for analyzing past threats and proactively defending against current ones. In this study, we propose an approach that leverages large language models (LLMs) to detect binary malware, specifically within JAR files, and utilizes the capabilities of LLMs combined with retrieval-augmented generation (RAG) to identify Common Vulnerabilities and Exposures (CVEs) that malware may exploit. We developed a proof-of-concept tool called MalCVE, which integrates binary code decompilation, deobfuscation, LLM-based code summarization, semantic similarity search, and CVE classification using LLMs. We evaluated MalCVE using a benchmark dataset of 3,839 JAR executables. MalCVE achieved a mean malware detection accuracy of 97%, at a fraction of the cost of commercial solutions. It is also the first tool to associate CVEs with binary malware, achieving a recall@10 of 65%, which is comparable to studies that perform similar analyses on source code.

摘要: 恶意软件攻击对经济的影响越来越大。商业恶意软件检测软件可能是昂贵的，并且很大程度上缺乏将恶意软件归因于其利用的特定软件漏洞的工具。了解恶意软件与其目标漏洞之间的联系对于分析过去的威胁并主动防御当前威胁至关重要。在这项研究中，我们提出了一种方法，利用大型语言模型（LLM）来检测二进制恶意软件，特别是在恶意软件文件中，并利用LLM的功能结合检索增强生成（RAG）来识别常见的漏洞和暴露（CVE），恶意软件可能会利用。我们开发了一个名为MalUTE的概念证明工具，它集成了二进制代码反编译、去模糊、基于LLM的代码摘要、语义相似性搜索和使用LLM的UTE分类。我们使用包含3，839个可执行文件的基准数据集评估了MalUTE。MalUTE的平均恶意软件检测准确率为97%，成本仅为商业解决方案的一小部分。它也是第一个将CVE与二进制恶意软件关联起来的工具，实现了65%的recall@10，与对源代码进行类似分析的研究相当。



## **3. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models**

SoK：大型语言模型中提示安全性的分类和评估 cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15476v1) [paper-pdf](http://arxiv.org/pdf/2510.15476v1)

**Authors**: Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing Huang, Yuan Hong

**Abstract**: Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date; and (5) presenting a comprehensive evaluation and leaderboard of state-of-the-art methods. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment.

摘要: 大型语言模型（LLM）已迅速成为现实世界应用程序的组成部分，为不同领域的服务提供动力。然而，它们的广泛部署暴露了严重的安全风险，特别是通过越狱提示，这些提示可能绕过模型对齐并引发有害输出。尽管对攻击和防御技术进行了深入的研究，但该领域仍然支离破碎：定义、威胁模型和评估标准差异很大，阻碍了系统性进步和公平比较。在知识系统化（SoK）中，我们通过以下方式解决这些挑战：（1）提出一种整体、多级别的分类法，组织LLM即时安全中的攻击、防御和漏洞;（2）将威胁模型和成本假设形式化为机器可读配置文件，以进行可重复的评估;（3）引入开源评估工具包，用于标准化、可审计的攻击和防御比较;（4）发布JAILCREAKDB，这是迄今为止最大的越狱和良性提示注释数据集;以及（5）提供最先进方法的全面评估和排行榜。我们的工作统一了碎片化的研究，为未来的研究提供了严格的基础，并支持开发适合高风险部署的稳健、值得信赖的LLM。



## **4. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CV

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15430v1) [paper-pdf](http://arxiv.org/pdf/2510.15430v1)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致更高的AUROC检测。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **5. DSSmoothing: Toward Certified Dataset Ownership Verification for Pre-trained Language Models via Dual-Space Smoothing**

DSA平滑：通过双空间平滑实现预训练语言模型的认证数据集所有权验证 cs.CR

13 pages, 21 figures

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15303v1) [paper-pdf](http://arxiv.org/pdf/2510.15303v1)

**Authors**: Ting Qiao, Xing Liu, Wenke Huang, Jianbin Li, Zhaoxin Fan, Yiming Li

**Abstract**: Large web-scale datasets have driven the rapid advancement of pre-trained language models (PLMs), but unauthorized data usage has raised serious copyright concerns. Existing dataset ownership verification (DOV) methods typically assume that watermarks remain stable during inference; however, this assumption often fails under natural noise and adversary-crafted perturbations. We propose the first certified dataset ownership verification method for PLMs based on dual-space smoothing (i.e., DSSmoothing). To address the challenges of text discreteness and semantic sensitivity, DSSmoothing introduces continuous perturbations in the embedding space to capture semantic robustness and applies controlled token reordering in the permutation space to capture sequential robustness. DSSmoothing consists of two stages: in the first stage, triggers are collaboratively embedded in both spaces to generate norm-constrained and robust watermarked datasets; in the second stage, randomized smoothing is applied in both spaces during verification to compute the watermark robustness (WR) of suspicious models and statistically compare it with the principal probability (PP) values of a set of benign models. Theoretically, DSSmoothing provides provable robustness guarantees for dataset ownership verification by ensuring that WR consistently exceeds PP under bounded dual-space perturbations. Extensive experiments on multiple representative web datasets demonstrate that DSSmoothing achieves stable and reliable verification performance and exhibits robustness against potential adaptive attacks.

摘要: 大型网络规模的数据集推动了预训练语言模型（PLM）的快速发展，但未经授权的数据使用引发了严重的版权问题。现有的数据集所有权验证（DOV）方法通常假设水印在推理过程中保持稳定;然而，这种假设在自然噪声和恶意干扰下往往会失败。我们提出了第一个基于双空间平滑的PLM认证数据集所有权验证方法（即，DSS平滑）。为了解决文本离散性和语义敏感性的挑战，DSSmooghing在嵌入空间中引入连续扰动以捕获语义稳健性，并在排列空间中应用受控令牌重新排序以捕获序列稳健性。DSA平滑由两个阶段组成：在第一阶段，触发器协作嵌入两个空间中，以生成受规范约束且稳健的水印数据集;在第二阶段，在验证期间在两个空间中应用随机平滑，以计算可疑模型的水印稳健性（WR），并将其与一组良性模型的主概率（PP）值进行统计比较。理论上，DSSmooting通过确保WR在有界双空间扰动下始终超过PP，为数据集所有权验证提供了可证明的稳健性保证。对多个代表性Web数据集的广泛实验表明，DSSmooghing实现了稳定可靠的验证性能，并对潜在的自适应攻击表现出鲁棒性。



## **6. PromptLocate: Localizing Prompt Injection Attacks**

Inbox Locate：本地化提示注入攻击 cs.CR

To appear in IEEE Symposium on Security and Privacy, 2026. For  slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.12252v2) [paper-pdf](http://arxiv.org/pdf/2510.12252v2)

**Authors**: Yuqi Jia, Yupei Liu, Zedian Shao, Jinyuan Jia, Neil Gong

**Abstract**: Prompt injection attacks deceive a large language model into completing an attacker-specified task instead of its intended task by contaminating its input data with an injected prompt, which consists of injected instruction(s) and data. Localizing the injected prompt within contaminated data is crucial for post-attack forensic analysis and data recovery. Despite its growing importance, prompt injection localization remains largely unexplored. In this work, we bridge this gap by proposing PromptLocate, the first method for localizing injected prompts. PromptLocate comprises three steps: (1) splitting the contaminated data into semantically coherent segments, (2) identifying segments contaminated by injected instructions, and (3) pinpointing segments contaminated by injected data. We show PromptLocate accurately localizes injected prompts across eight existing and eight adaptive attacks.

摘要: 提示注入攻击欺骗大型语言模型完成攻击者指定的任务，而不是其预期的任务，通过注入的提示污染其输入数据，其中包括注入的指令和数据。在受污染的数据中定位注入的提示对于攻击后的取证分析和数据恢复至关重要。尽管其重要性日益增加，但快速注射定位在很大程度上仍未被探索。在这项工作中，我们弥合这一差距，提出了本地化注入提示的第一种方法--martLocate。ObjectLocate包括三个步骤：（1）将受污染的数据拆分成语义一致的段，（2）识别被注入指令污染的段，以及（3）精确定位被注入数据污染的段。我们展示了EntLocate可以准确地定位跨越八种现有攻击和八种自适应攻击的注入提示。



## **7. WebInject: Prompt Injection Attack to Web Agents**

Web Injects：对Web代理的提示注入攻击 cs.LG

Appeared in EMNLP 2025 main conference. To better understand prompt  injection attacks, see https://people.duke.edu/~zg70/code/PromptInjection.pdf

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2505.11717v4) [paper-pdf](http://arxiv.org/pdf/2505.11717v4)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

摘要: 基于多模式大型语言模型（MLLM）的Web代理通过基于网页屏幕截图生成动作来与网页环境交互。在这项工作中，我们提出了WebInjects，这是一种提示注入攻击，它操纵网页环境以诱导Web代理执行攻击者指定的操作。我们的攻击对渲染网页的原始像素值添加了扰动。这些受干扰的像素被映射到屏幕截图后，扰动会导致Web代理执行攻击者指定的操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个数据集的广泛评估表明，WebInib非常有效，并且显着优于基线。



## **8. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs**

OCR-APT：使用子图异常检测和LLM从审计工作组重建APT故事 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15188v1) [paper-pdf](http://arxiv.org/pdf/2510.15188v1)

**Authors**: Ahmed Aly, Essam Mansour, Amr Youssef

**Abstract**: Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.

摘要: 高级持续威胁（APT）是一种隐秘的网络攻击，通常可以逃避系统级审计日志中的检测。起源图将这些日志建模为相连的实体和事件，揭示线性日志表示错过的关系。现有的系统将异常检测应用于这些图形，但通常存在高假阳性率和粗粒度警报的问题。它们对文件路径或IP等节点属性的依赖会导致虚假相关性，从而降低检测稳健性和可靠性。为了充分了解攻击的进展和影响，安全分析师需要能够对整个攻击生成准确、类似人类的叙述的系统。为了应对这些挑战，我们引入了OCR-APT，这是一种用于APT检测和重建类人攻击故事的系统。OCR-APT使用图神经网络（GNN）进行子图异常检测，学习节点周围的行为模式，而不是文件路径或IP等脆弱属性。这种方法可以实现更强大的异常检测。然后，它使用大型语言模型（LLM）对检测到的子图进行迭代，以重建多阶段攻击故事。每个阶段都在进行之前进行验证，减少幻觉并确保可解释的最终报告。我们对DARPA TC 3、OpTC和NODLINK数据集的评估表明，OCR-APT在检测准确性和警报可解释性方面都优于最先进的系统。此外，OCR-APT重建了全面捕捉攻击故事的人性化报告。



## **9. PoTS: Proof-of-Training-Steps for Backdoor Detection in Large Language Models**

PoTS：大型语言模型中后门检测的训练证明步骤 cs.CR

10 pages, 6 figures, 1 table. Accepted for presentation at FLLM 2025  (Vienna, Nov 2025)

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15106v1) [paper-pdf](http://arxiv.org/pdf/2510.15106v1)

**Authors**: Issam Seddik, Sami Souihi, Mohamed Tamaazousti, Sara Tucci Piergiovanni

**Abstract**: As Large Language Models (LLMs) gain traction across critical domains, ensuring secure and trustworthy training processes has become a major concern. Backdoor attacks, where malicious actors inject hidden triggers into training data, are particularly insidious and difficult to detect. Existing post-training verification solutions like Proof-of-Learning are impractical for LLMs due to their requirement for full retraining, lack of robustness against stealthy manipulations, and inability to provide early detection during training. Early detection would significantly reduce computational costs. To address these limitations, we introduce Proof-of-Training Steps, a verification protocol that enables an independent auditor (Alice) to confirm that an LLM developer (Bob) has followed the declared training recipe, including data batches, architecture, and hyperparameters. By analyzing the sensitivity of the LLMs' language modeling head (LM-Head) to input perturbations, our method can expose subtle backdoor injections or deviations in training. Even with backdoor triggers in up to 10 percent of the training data, our protocol significantly reduces the attacker's ability to achieve a high attack success rate (ASR). Our method enables early detection of attacks at the injection step, with verification steps being 3x faster than training steps. Our results highlight the protocol's potential to enhance the accountability and security of LLM development, especially against insider threats.

摘要: 随着大型语言模型（LLM）在关键领域获得关注，确保安全且值得信赖的培训流程已成为一个主要问题。后门攻击（恶意行为者将隐藏的触发器注入训练数据）特别阴险且难以检测。现有的训练后验证解决方案（如学习证明）对于LLM来说是不切实际的，因为它们需要完全再培训、缺乏针对隐形操纵的鲁棒性以及无法在训练期间提供早期检测。早期检测将显着降低计算成本。为了解决这些限制，我们引入了训练证明步骤，这是一种验证协议，使独立审计员（Alice）能够确认LLM开发人员（Bob）是否遵循了声明的训练配方，包括数据批处理、架构和超参数。通过分析LLM的语言建模头（LM-Head）对输入扰动的敏感性，我们的方法可以暴露训练中微妙的后门注入或偏差。即使在高达10%的训练数据中存在后门触发，我们的协议也会显着降低攻击者实现高攻击成功率（ASB）的能力。我们的方法可以在注入步骤早期检测攻击，验证步骤比训练步骤快3倍。我们的结果凸显了该协议在增强LLM开发的问责制和安全性方面的潜力，特别是针对内部威胁。



## **10. Sequential Comics for Jailbreaking Multimodal Large Language Models via Structured Visual Storytelling**

通过结构化视觉讲故事破解多模式大型语言模型的连续漫画 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15068v1) [paper-pdf](http://arxiv.org/pdf/2510.15068v1)

**Authors**: Deyue Zhang, Dongdong Yang, Junjie Mu, Quancheng Zou, Zonghao Ying, Wenzhuo Xu, Zhao Liu, Xuan Wang, Xiangzheng Zhang

**Abstract**: Multimodal large language models (MLLMs) exhibit remarkable capabilities but remain susceptible to jailbreak attacks exploiting cross-modal vulnerabilities. In this work, we introduce a novel method that leverages sequential comic-style visual narratives to circumvent safety alignments in state-of-the-art MLLMs. Our method decomposes malicious queries into visually innocuous storytelling elements using an auxiliary LLM, generates corresponding image sequences through diffusion models, and exploits the models' reliance on narrative coherence to elicit harmful outputs. Extensive experiments on harmful textual queries from established safety benchmarks show that our approach achieves an average attack success rate of 83.5\%, surpassing prior state-of-the-art by 46\%. Compared with existing visual jailbreak methods, our sequential narrative strategy demonstrates superior effectiveness across diverse categories of harmful content. We further analyze attack patterns, uncover key vulnerability factors in multimodal safety mechanisms, and evaluate the limitations of current defense strategies against narrative-driven attacks, revealing significant gaps in existing protections.

摘要: 多模式大型语言模型（MLLM）表现出非凡的能力，但仍然容易受到利用跨模式漏洞的越狱攻击。在这项工作中，我们引入了一种新颖的方法，该方法利用连续漫画风格的视觉叙事来规避最先进的MLLM中的安全对齐。我们的方法使用辅助LLM将恶意查询分解为视觉上无害的讲故事元素，通过扩散模型生成相应的图像序列，并利用模型对叙事连贯性的依赖来引发有害输出。对已建立的安全基准中的有害文本查询进行的广泛实验表明，我们的方法的平均攻击成功率为83.5%，比之前的最先进技术高出46%。与现有的视觉越狱方法相比，我们的顺序叙事策略在不同类别的有害内容中表现出卓越的有效性。我们进一步分析攻击模式，揭示多模式安全机制中的关键脆弱性因素，并评估当前防御策略针对叙事驱动攻击的局限性，揭示现有保护中的显着差距。



## **11. Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers**

保持冷静并避免有害内容：概念一致和潜在操纵以获得更安全的答案 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.12672v2) [paper-pdf](http://arxiv.org/pdf/2510.12672v2)

**Authors**: Ruben Belo, Marta Guimaraes, Claudia Soares

**Abstract**: Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation CALM, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging concept whitening technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference.

摘要: 大型语言模型容易受到绕过内置安全护栏的越狱攻击（例如，通过用对抗性提示欺骗模型）。我们提出了概念对齐和概念操纵CALM，这是一种推理时方法，通过修改模型最后一层的潜在表示来抑制有害概念，而无需重新训练。CALM利用计算机视觉的概念白化技术与垂直投影相结合，删除了与有害内容相关的不需要的潜在方向，同时保留了模型性能。实验表明，CALM减少了有害输出，并在大多数指标上优于基线方法，为人工智能安全提供了一种轻量级方法，无需额外的训练数据或模型微调，同时在推理时只产生很小的计算负担。



## **12. Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks**

主动蜜罐保护系统：探测和识别多回合LLM越狱 cs.CR

6pages, 2 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15017v1) [paper-pdf](http://arxiv.org/pdf/2510.15017v1)

**Authors**: ChenYu Wu, Yi Wang, Yang Liao

**Abstract**: Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.

摘要: 大型语言模型（LLM）越来越容易受到多回合越狱攻击，对手会反复引发绕过单回合安全过滤器的有害行为。现有的防御主要依赖于被动拒绝，这要么无法抵御自适应攻击者，要么过度限制良性用户。我们提出了一种基于蜜罐的主动护栏系统，将风险规避转化为风险利用。我们的框架微调了诱饵模型，以生成模棱两可、不可操作但语义相关的响应，这些响应作为试探用户意图的诱饵。结合受保护的LLM的安全回复，系统插入主动诱饵问题，通过多回合交互逐渐暴露恶意意图。我们进一步引入了蜜罐效用评分（HUS），衡量诱饵反应的吸引力和可行性，并使用防御效能率（BER）来平衡安全性和可用性。针对GPT-4 o最近攻击方法的MTJ数据集的初步实验表明，我们的系统在保留良性用户体验的同时，显着破坏了越狱成功。



## **13. Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs**

机器放弃学习通过对LLM的约束干预来满足对抗鲁棒性 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.03567v3) [paper-pdf](http://arxiv.org/pdf/2510.03567v3)

**Authors**: Fatmazohra Rezkellah, Ramzi Dakhmouche

**Abstract**: With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.

摘要: 随着大型语言模型（LLM）的日益采用，需要更多的定制来确保隐私保护和安全生成。我们从两个关键方面实现这一目标：忘记敏感信息和对越狱攻击的稳健性。我们研究了各种受约束的优化公式，以\r {统一方式}解决这两个方面，通过找到对LLM权重的最小可能干预，这些干预要么使给定的词汇集不可达，要么通过将部分权重转移到\r {更安全}区域来嵌入LLM，对定制攻击具有鲁棒性。除了统一两个关键属性之外，这种方法与之前的工作形成鲜明对比，因为它不需要通常不可用或代表计算负担的Oracle分类器。令人惊讶的是，我们发现我们提出的最简单的逐点基于约束的干预比最大-最小干预具有更好的性能，同时具有更低的计算成本。与最先进的防御方法的比较表明了所提出的方法的优越性能。



## **14. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

大型语言模型中对偏见激发的对抗鲁棒性进行基准测试：利用LLM作为评委的可扩展自动化评估 cs.CL

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2504.07887v2) [paper-pdf](http://arxiv.org/pdf/2504.07887v2)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.

摘要: 大型语言模型（LLM）日益融入关键社会领域，引发了人们对嵌入式偏见的担忧，这些偏见可能会延续刻板印象并破坏公平性。此类偏见可能源于训练数据的历史不平等、语言不平衡或对抗操纵。尽管采取了缓解措施，但最近的研究表明，LLM仍然容易受到引发偏见输出的对抗攻击。这项工作提出了一个可扩展的基准框架来评估LLM对对抗性偏见引发的稳健性。我们的方法包括：（i）针对不同的社会文化偏见，系统地探索跨多项任务的模型，（ii）使用LLM作为法官的方法通过安全评分量化稳健性，以及（iii）采用越狱技术来揭示安全漏洞。为了促进系统性基准测试，我们发布了一个精心策划的偏差相关提示数据集，名为ClearAR-Bias。我们的分析将DeepSeek V3确定为最可靠的LLM法官，揭示了偏见复原力是不平衡的，其中年龄、残疾和交叉偏见是最突出的。一些小型模型在安全性方面优于大型模型，这表明培训和架构可能比规模更重要。然而，没有一个模型对对抗性诱导完全稳健，事实证明，使用低资源语言或拒绝抑制的越狱攻击在模型家族中都有效。我们还发现，连续几代LLM表现出轻微的安全性收益，而针对医疗领域微调的模型往往不如通用模型安全。



## **15. ATGen: Adversarial Reinforcement Learning for Test Case Generation**

ATGen：用于测试用例生成的对抗强化学习 cs.SE

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14635v1) [paper-pdf](http://arxiv.org/pdf/2510.14635v1)

**Authors**: Qingyao Li, Xinyi Dai, Weiwen Liu, Xiangyang Li, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang

**Abstract**: Large Language Models (LLMs) excel at code generation, yet their outputs often contain subtle bugs, for which effective test cases are a critical bottleneck. Existing test generation methods, whether based on prompting or supervised fine-tuning, rely on static datasets. This imposes a ``fixed-difficulty ceiling'', fundamentally limiting their ability to uncover novel or more complex bugs beyond their training scope. To overcome this, we introduce ATGen, a framework that trains a test case generator via adversarial reinforcement learning. ATGen pits a test generator against an adversarial code generator that continuously crafts harder bugs to evade the current policy. This dynamic loop creates a curriculum of increasing difficulty challenging current policy. The test generator is optimized via Reinforcement Learning (RL) to jointly maximize ``Output Accuracy'' and ``Attack Success'', enabling it to learn a progressively stronger policy that breaks the fixed-difficulty ceiling of static training. Extensive experiments demonstrate that ATGen significantly outperforms state-of-the-art baselines. We further validate its practical utility, showing it serves as both a more effective filter for Best-of-N inference and a higher-quality reward source for training code generation models. Our work establishes a new, dynamic paradigm for improving the reliability of LLM-generated code.

摘要: 大型语言模型（LLM）擅长代码生成，但它们的输出通常包含微妙的bug，有效的测试用例是一个关键的瓶颈。现有的测试生成方法，无论是基于提示还是监督微调，都依赖于静态数据集。这强加了一个“固定难度上限”，从根本上限制了他们发现超出其培训范围的新的或更复杂的错误的能力。为了克服这一点，我们引入了ATGen，这是一个通过对抗性强化学习来训练测试用例生成器的框架。ATGen将测试生成器与对抗代码生成器进行了比较，后者不断地制造更难的错误来逃避当前政策。这种动态循环创造了挑战当前政策难度不断增加的课程。测试生成器通过强化学习（RL）进行优化，以共同最大化“输出准确度”和“攻击时间”，使其能够学习逐渐更强的策略，打破静态训练的固定难度上限。大量实验表明，ATGen的性能显着优于最先进的基线。我们进一步验证了它的实际实用性，表明它既是N最佳推理的更有效过滤器，也是训练代码生成模型的更高质量的奖励来源。我们的工作建立了一个新的动态范式，用于提高LLM生成的代码的可靠性。



## **16. Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses**

Checkpoint-GCG：审计和攻击基于微调的提示注入防御 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2505.15738v2) [paper-pdf](http://arxiv.org/pdf/2505.15738v2)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world applications ranging from chatbots to agentic systems, where they are expected to process untrusted data and follow trusted instructions. Failure to distinguish between the two poses significant security risks, exploited by prompt injection attacks, which inject malicious instructions into the data to control model outputs. Model-level defenses have been proposed to mitigate prompt injection attacks. These defenses fine-tune LLMs to ignore injected instructions in untrusted data. We introduce Checkpoint-GCG, a white-box attack against fine-tuning-based defenses. Checkpoint-GCG enhances the Greedy Coordinate Gradient (GCG) attack by leveraging intermediate model checkpoints produced during fine-tuning to initialize GCG, with each checkpoint acting as a stepping stone for the next one to continuously improve attacks. First, we instantiate Checkpoint-GCG to evaluate the robustness of the state-of-the-art defenses in an auditing setup, assuming both (a) full knowledge of the model input and (b) access to intermediate model checkpoints. We show Checkpoint-GCG to achieve up to $96\%$ attack success rate (ASR) against the strongest defense. Second, we relax the first assumption by searching for a universal suffix that would work on unseen inputs, and obtain up to $89.9\%$ ASR against the strongest defense. Finally, we relax both assumptions by searching for a universal suffix that would transfer to similar black-box models and defenses, achieving an ASR of $63.9\%$ against a newly released defended model from Meta.

摘要: 大型语言模型（LLM）越来越多地部署在从聊天机器人到代理系统的现实世界应用程序中，它们预计能够处理不受信任的数据并遵循受信任的指令。未能区分两者会带来巨大的安全风险，这些风险被即时注入攻击所利用，这些攻击将恶意指令注入到数据中以控制模型输出。已经提出了模型级防御来缓解即时注入攻击。这些防御措施对LLM进行微调，以忽略不受信任数据中注入的指令。我们引入Checkpoint-GCG，这是一种针对基于微调的防御的白盒攻击。Checkpoint-GCG通过利用微调期间产生的中间模型检查点来初始化GCG来增强贪婪协调梯度（GCG）攻击，每个检查点都充当下一个检查点的垫脚石，以持续改进攻击。首先，我们实例化Checkpoint-GCG，以评估审计设置中最先进防御的稳健性，假设（a）完全了解模型输入和（b）访问中间模型检查点。我们表明Checkpoint-GCG可以在最强防御下实现高达96%美元的攻击成功率（ASB）。其次，我们通过搜索适用于不可见输入的通用后缀来放宽第一个假设，并针对最强防御获得高达89.9美元\%$ASB。最后，我们通过搜索一个通用后缀来放宽这两个假设，该后缀将转移到类似的黑匣子模型和防御，与Meta新发布的防御模型相比，实现了63.9美元的ASB。



## **17. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

SoK：评估大型语言模型的越狱护栏 cs.CR

Accepted by IEEE S&P 2026 Cycle 1

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2506.10597v2) [paper-pdf](http://arxiv.org/pdf/2506.10597v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety alignments. Guardrails--external defense mechanisms that monitor and control LLM interactions--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, provide insights into optimizing their defense mechanisms, and explore their universality across attack types. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.

摘要: 大型语言模型（LLM）取得了显着的进步，但它们的部署暴露了关键漏洞，特别是规避安全一致的越狱攻击。Guardrails--监控和控制LLM交互的外部防御机制--已成为一种有希望的解决方案。然而，目前LLM护栏格局支离破碎，缺乏统一的分类和全面的评估框架。在这篇知识系统化（SoK）论文中，我们首次对LLM的越狱护栏进行了整体分析。我们提出了一种新颖的多维分类法，根据六个关键维度对护栏进行分类，并引入安全-效率-效用评估框架来评估其实际有效性。通过广泛的分析和实验，我们确定了现有护栏方法的优点和局限性，提供优化其防御机制的见解，并探索其在攻击类型中的普遍性。我们的工作为未来的研究和开发提供了结构化的基础，旨在指导强大的LLM护栏的有原则的推进和部署。该代码可在https://github.com/xunguangwang/SoK4JailbreakGuardrails上获取。



## **18. SPIRIT: Patching Speech Language Models against Jailbreak Attacks**

精神：修补语音语言模型以防止越狱攻击 eess.AS

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2505.13541v2) [paper-pdf](http://arxiv.org/pdf/2505.13541v2)

**Authors**: Amirbek Djanibekov, Nurdaulet Mukhituly, Kentaro Inui, Hanan Aldarmaki, Nils Lukas

**Abstract**: Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100% attack success rate in some instances. To improve security, we propose post-hoc patching defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.

摘要: 语音语言模型（SLC）通过口头指令实现自然交互，通过检测语音中的细微差别更有效地捕捉用户意图。与基于文本的模型相比，更丰富的语音信号会带来新的安全风险，因为对手可以通过向语音注入难以感知的噪音来更好地绕过安全机制。我们分析了对抗性攻击，发现STM更容易受到越狱攻击，在某些情况下可以实现完美的100%攻击成功率。为了提高安全性，我们提出了事后修补防御，用于通过修改SPL的激活来在推理期间进行干预，从而将稳健性提高高达99%，并且（i）对效用的影响可以忽略不计，并且（ii）无需任何重新训练。我们进行消融研究，以最大限度地提高防御的功效并改善实用性/安全性权衡，并通过SLS特有的大规模基准进行验证。



## **19. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

Lexo：通过LLM辅助程序再生消除隐形供应链攻击 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14522v1) [paper-pdf](http://arxiv.org/pdf/2510.14522v1)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.

摘要: 软件供应链攻击是开源软件生态系统中一个重要且持续存在的问题。这些攻击保留了组件实现的标准功能，但还隐藏了仅在组件到达其目标环境时激活的恶意功能。Lexo通过自动学习和重新生成潜在恶意组件的无可识别性版本来解决此类隐形攻击。Lexo首先生成一组输入-输出对来建模组件的完整可观察行为，然后使用其合成原始组件的新版本。新组件实现了原始功能，但避免了隐蔽的恶意行为。在整个重建过程中，Lexo会咨询大型语言模型（LLM）的几个不同实例，使用正确性和覆盖率指标来引导这些实例，并保护它们的结果。我们对100多个现实世界的包（包括高调的隐形供应链攻击）的评估表明，Lexo可以跨多个域扩展，有效地再生代码（平均<100），保持兼容性，并成功消除了几个现实世界的供应链攻击中的恶意代码，即使在最先进的LLM在提示时未能消除恶意代码的情况下也是如此。



## **20. Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers**

我的优化预算是否受到影响？探索基于LLM的优化器的漏洞 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14381v1) [paper-pdf](http://arxiv.org/pdf/2510.14381v1)

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes

**Abstract**: Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.

摘要: 大型语言模型（LLM）系统现在支撑着聊天机器人、计算机使用助理和自主机器人等日常人工智能应用程序，其中的性能通常取决于精心设计的提示。基于LLM的提示优化器通过迭代地从评分反馈中改进提示来减少这一工作，但该优化阶段的安全性仍然不足。我们在基于LLM的即时优化中首次对中毒风险进行了系统分析。使用HarmBench，我们发现系统更容易受到操纵反馈的影响，而不是注入查询的影响：基于反馈的攻击将攻击成功率（ASB）提高高达$\Delta$ASO = 0.48。我们引入了一种简单的虚假奖励攻击，它不需要访问奖励模型并显着增加了漏洞，我们提出了一种轻量级的突出显示防御，可以将虚假奖励$\Delta$ASB从0.23减少到0.07，而不会降低效用。这些结果将即时优化管道建立为一流的攻击面，并激励反馈渠道和优化框架更强有力的保障措施。



## **21. CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment**

CoreGuard：保护LLM的基础能力，防止边缘部署中的模型窃取 cs.CR

Accepted by NeurIPS 2025 Conference

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2410.13903v2) [paper-pdf](http://arxiv.org/pdf/2410.13903v2)

**Authors**: Qinfeng Li, Tianyue Luo, Xuhong Zhang, Yangfan Xie, Zhiqiang Shen, Lijun Zhang, Yier Jin, Hao Peng, Xinkui Zhao, Xianwei Zhu, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) exhibit strong generalization capabilities across diverse tasks and are increasingly deployed on edge devices for efficiency and privacy reasons. However, deploying proprietary LLMs at the edge without adequate protection introduces critical security threats. Attackers can extract model weights and architectures, enabling unauthorized copying and misuse. Even when protective measures prevent full extraction of model weights, attackers may still perform advanced attacks, such as fine-tuning, to further exploit the model. Existing defenses against these threats typically incur significant computational and communication overhead, making them impractical for edge deployment. To safeguard the edge-deployed LLMs, we introduce CoreGuard, a computation- and communication-efficient protection method. CoreGuard employs an efficient protection protocol to reduce computational overhead and minimize communication overhead via a propagation protocol. Extensive experiments show that CoreGuard achieves upper-bound security protection with negligible overhead.

摘要: 专有大型语言模型（LLM）在不同任务中表现出强大的概括能力，并且出于效率和隐私原因越来越多地部署在边缘设备上。然而，在没有充分保护的情况下在边缘部署专有LLM会带来严重的安全威胁。攻击者可以提取模型权重和架构，从而实现未经授权的复制和滥用。即使保护措施阻止完全提取模型权重，攻击者仍然可能执行高级攻击（例如微调）以进一步利用模型。针对这些威胁的现有防御通常会产生大量的计算和通信负担，使其对于边缘部署来说不切实际。为了保护边缘部署的LLM，我们引入了CoreGuard，这是一种计算和通信高效的保护方法。CoreGuard采用高效的保护协议来减少计算负担并通过传播协议最大限度地减少通信负担。大量实验表明，CoreGuard以可忽略不计的费用即可实现上限安全保护。



## **22. When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment**

当风格破坏安全性时：保护LLM免受表面风格一致 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2506.07452v2) [paper-pdf](http://arxiv.org/pdf/2506.07452v2)

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.

摘要: 大型语言模型（LLM）可以用特定的风格提示（例如，将响应格式化为列表），包括恶意查询中。之前的越狱研究主要通过额外的字符串转换来增强这些查询，以最大限度地提高攻击成功率（ASB）。然而，原始查询中与恶意意图语义无关的样式模式的影响仍不清楚。在这项工作中，我们试图了解风格模式是否会损害LLM的安全性，肤浅的风格对齐如何增加模型的脆弱性，以及如何在对齐过程中最好地减轻这些风险。我们首先将ASR膨胀定义为由于现有越狱基准查询中的样式模式而导致的ASR增加。通过评估7个基准测试中的32个LLM，我们发现几乎所有模型都表现出ASR膨胀。值得注意的是，通货膨胀与LLM对风格模式的相对关注相关，当通货膨胀发生时，这也与其预防调整数据重叠更多。然后，我们调查了表面的风格调整，发现对特定风格的微调使LLM更容易受到相同风格的越狱的影响。最后，我们提出了SafeStyle，这是一种防御策略，它结合了少量的安全训练数据，经过扩展以匹配微调数据中风格模式的分布。在三个LLM、六个微调风格设置和两个现实世界的描述调整数据集中，SafeStyle在维护LLM安全方面始终优于基线。



## **23. Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies**

Terrarium：重新审视多智能体安全、隐私和安全研究黑板 cs.AI

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14312v1) [paper-pdf](http://arxiv.org/pdf/2510.14312v1)

**Authors**: Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian

**Abstract**: A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems.

摘要: 由大型语言模型（LLM）支持的多代理系统（MAS）可以自动化繁琐的用户任务，例如需要代理间协作的会议安排。LLM支持细致入微的协议，可考虑非结构化私人数据、用户约束和偏好。然而，这种设计引入了新的风险，包括未对准和恶意方的攻击，从而危及代理或窃取用户数据。在本文中，我们提出了Terrarium框架，用于对基于LLM的MAS中的安全性、隐私和安全性进行细粒度研究。我们重新设计的黑板设计，在多智能体系统的早期方法，创建一个模块化的，可配置的多智能体协作的测试床。我们确定关键的攻击向量，如错位，恶意代理，受损的通信和数据中毒。我们实现了三个协作MAS的情况下，四个代表性的攻击，以证明框架的灵活性。通过提供快速原型化、评估和验证防御和设计的工具，Terrarium旨在加速向可信赖的多智能体系统发展。



## **24. RHINO: Guided Reasoning for Mapping Network Logs to Adversarial Tactics and Techniques with Large Language Models**

RHINO：将网络条件映射到具有大型语言模型的对抗策略和技术的引导推理 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14233v1) [paper-pdf](http://arxiv.org/pdf/2510.14233v1)

**Authors**: Fanchao Meng, Jiaping Gui, Yunbo Li, Yue Wu

**Abstract**: Modern Network Intrusion Detection Systems generate vast volumes of low-level alerts, yet these outputs remain semantically fragmented, requiring labor-intensive manual correlation with high-level adversarial behaviors. Existing solutions for automating this mapping-rule-based systems and machine learning classifiers-suffer from critical limitations: rule-based approaches fail to adapt to novel attack variations, while machine learning methods lack contextual awareness and treat tactic-technique mapping as a syntactic matching problem rather than a reasoning task. Although Large Language Models have shown promise in cybersecurity tasks, preliminary experiments reveal that existing LLM-based methods frequently hallucinate technique names or produce decontextualized mappings due to their single-step classification approach.   To address these challenges, we introduce RHINO, a novel framework that decomposes LLM-based attack analysis into three interpretable phases mirroring human reasoning: (1) behavioral abstraction, where raw logs are translated into contextualized narratives; (2) multi-role collaborative inference, generating candidate techniques by evaluating behavioral evidence against MITRE ATT&CK knowledge; and (3) validation, cross-referencing predictions with official MITRE definitions to rectify hallucinations. RHINO bridges the semantic gap between low-level observations and adversarial intent while improving output reliability through structured reasoning.   We evaluate RHINO on three benchmarks across four backbone models. RHINO achieved high accuracy, with model performance ranging from 86.38% to 88.45%, resulting in relative gains from 24.25% to 76.50% across different models. Our results demonstrate that RHINO significantly enhances the interpretability and scalability of threat analysis, offering a blueprint for deploying LLMs in operational security settings.

摘要: 现代网络入侵检测系统会生成大量低级警报，但这些输出仍然是语义碎片化的，需要与高级对抗行为进行劳动密集型的手动关联。用于自动化这种映射的现有解决方案--基于规则的系统和机器学习分类器--存在严重的局限性：基于规则的方法无法适应新颖的攻击变体，而机器学习方法缺乏上下文感知，并将战术技术映射视为语法匹配问题而不是推理任务。尽管大型语言模型在网络安全任务中表现出了希望，但初步实验表明，现有的基于LLM的方法由于其分步分类方法而经常使技术名称产生幻觉或产生去上下文化映射。   为了应对这些挑战，我们引入了RHINO，这是一个新颖的框架，它将基于LLM的攻击分析分解为反映人类推理的三个可解释阶段：（1）行为抽象，其中原始日志被翻译为上下文化叙述;（2）多角色协作推理，通过针对MITRE ATT & CK知识评估行为证据来生成候选技术;以及（3）验证，将预测与官方MITRE定义交叉引用以纠正幻觉。RHINO弥合了低级观察和对抗意图之间的语义差距，同时通过结构化推理提高了输出的可靠性。   我们对四种主干模型的三个基准进行了评估。RHINO实现了高准确度，模型性能范围为86.38%至88.45%，不同型号的相对收益范围为24.25%至76.50%。我们的结果表明，RHINO显着增强了威胁分析的可解释性和可扩展性，为在运营安全环境中部署LLM提供了蓝图。



## **25. Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks**

人类恶意在特工中的回响：针对多轮在线骚扰攻击对标LLM cs.AI

13 pages, 4 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14207v1) [paper-pdf](http://arxiv.org/pdf/2510.14207v1)

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.

摘要: 大型语言模型（LLM）代理正在为越来越多的交互式Web应用程序提供支持，但仍然容易受到滥用和伤害。之前的越狱研究主要集中在单回合提示上，而真正的骚扰通常在多回合互动中展开。在这项工作中，我们提出了在线骚扰统计基准，包括：（i）合成的多回合骚扰对话数据集，（ii）多代理（例如，骚扰者、受害者）由重复博弈理论指导的模拟，（iii）跨越记忆、规划和微调攻击代理的三种越狱方法，以及（iv）混合方法评估框架。我们利用两个突出的LLM，LLaMA-3.1-8B-Instruct（开源）和Gemini-2.0-flash（闭源）。我们的研究结果表明，越狱调整使骚扰几乎可以保证，在Llama中，攻击成功率为95.78- 96.89% vs. 57.25- 64.19%，在Gemini中，攻击成功率为99.33% vs. 98.46%，同时在两种模型中，拒绝率都大幅降低到1-2%。最普遍的有毒行为是侮辱，84.9- 87.8%对44.2- 50.8%，没有调整，81.2- 85.1%对31.5- 38.8%，表明与敏感类别相比，如性或种族骚扰，护栏较弱。定性评估进一步表明，攻击代理再现人类一样的侵略配置文件，如马基雅维利/心理变态模式下的计划，和自恋倾向的记忆。与直觉相反，闭源模型和开源模型在不同时期表现出不同的升级轨迹，而闭源模型则表现出显着的脆弱性。总体而言，我们的研究结果表明，多回合和基于理论的攻击不仅能够高成功率，而且还模拟了类人的骚扰动态，推动了强大的安全护栏的开发，以最终确保在线平台的安全和负责任。



## **26. One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery**

一个错误，数百个落后：用于大规模错误发现的LLM cs.SE

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.14036v1) [paper-pdf](http://arxiv.org/pdf/2510.14036v1)

**Authors**: Qiushi Wu, Yue Xiao, Dhilung Kirat, Kevin Eykholt, Jiyong Jang, Douglas Lee Schales

**Abstract**: Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program.   In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset.

摘要: 修复大型程序中的错误是一项具有挑战性的任务，需要大量的时间和精力。一旦发现bug，它会报告给项目维护人员，他们与报告者一起修复它，并最终关闭问题。然而，在整个程序中，通常有类似的代码段，这些代码段也可能包含错误，但在发现过程中被遗漏了。单独查找和修复每个重复出现的bug实例是劳动密集型的。更令人担忧的是，错误报告可能会无意中扩大攻击面，因为它们为攻击者提供了一种可利用的模式，而这种模式在程序的其他部分中可能无法解决。   在本文中，我们探讨了这些重复出现的模式错误（RPB），它们在程序的各个代码段中甚至不同程序中重复出现，源于相同的根本原因，但尚未解决。我们的调查表明，RPB很普遍，并且会严重损害软件程序的安全性。本文介绍了BugStone，这是一个由LLVM和大型语言模型（LLM）支持的程序分析系统。关键的观察是，许多RPB都有一个补丁实例，可以利用它来识别一致的错误模式，例如特定的API滥用。通过检查整个程序的这种模式，可以识别可能容易受到攻击的类似代码部分。BugStone从135个独特的RPB开始，在Linux内核中发现了超过22，000个新的潜在问题。对其中400项发现进行了手动分析，证实了246项有效。我们还根据最近23个顶级会议作品报告的超过19，000个安全漏洞创建了一个数据集。我们手动注释数据集，识别80个重复出现的模式和850个相应的修复。即使选择了具有成本效益的模型，BugStone也在数据集上实现了92.2%的精确度和79.1%的成对准确度。



## **27. Signature in Code Backdoor Detection, how far are we?**

代码后门检测签名，我们还走多远？ cs.SE

20 pages, 3 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13992v1) [paper-pdf](http://arxiv.org/pdf/2510.13992v1)

**Authors**: Quoc Hung Le, Thanh Le-Cong, Bach Le, Bowen Xu

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into software development workflows, they also become prime targets for adversarial attacks. Among these, backdoor attacks are a significant threat, allowing attackers to manipulate model outputs through hidden triggers embedded in training data. Detecting such backdoors remains a challenge, and one promising approach is the use of Spectral Signature defense methods that identify poisoned data by analyzing feature representations through eigenvectors. While some prior works have explored Spectral Signatures for backdoor detection in neural networks, recent studies suggest that these methods may not be optimally effective for code models. In this paper, we revisit the applicability of Spectral Signature-based defenses in the context of backdoor attacks on code models. We systematically evaluate their effectiveness under various attack scenarios and defense configurations, analyzing their strengths and limitations. We found that the widely used setting of Spectral Signature in code backdoor detection is often suboptimal. Hence, we explored the impact of different settings of the key factors. We discovered a new proxy metric that can more accurately estimate the actual performance of Spectral Signature without model retraining after the defense.

摘要: 随着大型语言模型（LLM）越来越多地集成到软件开发工作流程中，它们也成为对抗性攻击的主要目标。其中，后门攻击是一个重大威胁，允许攻击者通过嵌入训练数据中的隐藏触发器操纵模型输出。检测此类后门仍然是一个挑战，一种有前途的方法是使用光谱签名防御方法，该方法通过特征载体分析特征表示来识别有毒数据。虽然之前的一些工作已经探索了用于神经网络后门检测的光谱签名，但最近的研究表明，这些方法对于代码模型可能不是最佳有效的。在本文中，我们重新审视了基于光谱签名的防御在代码模型后门攻击的背景下的适用性。我们系统地评估它们在各种攻击场景和防御配置下的有效性，分析它们的优势和局限性。我们发现，代码后门检测中广泛使用的光谱签名设置通常是次优的。因此，我们探讨了不同设置对关键因素的影响。我们发现了一种新的代理指标，它可以更准确地估计Spectral Signature的实际性能，而无需在防御后进行模型重新训练。



## **28. LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks**

基于LLM的在上下文中学习，用于无人机辅助传感器网络中的数据收集调度 cs.AI

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2504.14556v2) [paper-pdf](http://arxiv.org/pdf/2504.14556v2)

**Authors**: Yousef Emami, Hao Zhou, SeyedSina Nabavirazani, Luis Almeida

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly being utilized in various private and commercial applications, e.g., traffic control, parcel delivery, and Search and Rescue (SAR) missions. Machine Learning (ML) methods used in UAV-Assisted Sensor Networks (UASNETs) and, especially, in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sampling efficiency, which conflict with the urgency of emergencies, such as SAR missions. In this paper, an In-Context Learning (ICL)-Data Collection Scheduling (ICLDC) system is proposed as an alternative to DRL in emergencies. The UAV collects sensory data and transmits it to a Large Language Model (LLM), which creates a task description in natural language. From this description, the UAV receives a data collection schedule that must be executed. A verifier ensures safe UAV operations by evaluating the schedules generated by the LLM and overriding unsafe schedules based on predefined rules. The system continuously adapts by incorporating feedback into the task descriptions and using this for future decisions. This method is tested against jailbreaking attacks, where the task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC significantly reduces cumulative packet loss compared to both the DQN and Maximum Channel Gain baselines. ICLDC presents a promising direction for intelligent scheduling and control in UASNETs.

摘要: 无人驾驶飞行器（UF）越来越多地用于各种私人和商业应用，例如交通管制、包裹递送和搜救（SAR）任务。无人机辅助传感器网络（UASNET），特别是深度强化学习（DRL）中使用的机器学习（ML）方法面临着复杂且冗长的模型训练、模拟与现实之间的差距以及低采样效率等挑战，这些挑战与紧急情况的紧迫性相冲突，例如SAR任务。本文提出了一种上下文学习（ICL）-数据收集调度（ICLDC）系统作为紧急情况下DRL的替代方案。无人机收集传感数据并将其传输到大型语言模型（LLM），该模型以自然语言创建任务描述。根据此描述，无人机收到必须执行的数据收集计划。验证器通过评估LLM生成的计划并根据预定义的规则覆盖不安全的计划来确保无人机操作的安全。该系统通过将反馈纳入任务描述并将其用于未来的决策来不断进行调整。该方法经过针对越狱攻击的测试，其中任务描述被操纵以破坏网络性能，凸显了LLM对此类攻击的脆弱性。与DQN和最大通道收益基线相比，拟议的ICLDC显着减少了累积数据包丢失。ICLDC为UASNET的智能调度和控制提供了一个有前途的方向。



## **29. In-Browser LLM-Guided Fuzzing for Real-Time Prompt Injection Testing in Agentic AI Browsers**

浏览器内LLM引导的模糊处理，用于在大型AI浏览器中进行实时提示注入测试 cs.CR

37 pages , 10 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13543v1) [paper-pdf](http://arxiv.org/pdf/2510.13543v1)

**Authors**: Avihay Cohen

**Abstract**: Large Language Model (LLM) based agents integrated into web browsers (often called agentic AI browsers) offer powerful automation of web tasks. However, they are vulnerable to indirect prompt injection attacks, where malicious instructions hidden in a webpage deceive the agent into unwanted actions. These attacks can bypass traditional web security boundaries, as the AI agent operates with the user privileges across sites. In this paper, we present a novel fuzzing framework that runs entirely in the browser and is guided by an LLM to automatically discover such prompt injection vulnerabilities in real time.

摘要: 集成到Web浏览器（通常称为代理AI浏览器）中的基于大型语言模型（LLM）的代理提供了强大的Web任务自动化。然而，它们很容易受到间接提示注入攻击，即隐藏在网页中的恶意指令欺骗代理采取不必要的操作。这些攻击可以绕过传统的网络安全边界，因为AI代理以跨网站的用户特权运行。在本文中，我们提出了一种新颖的模糊框架，该框架完全在浏览器中运行，并在LLM的指导下自动实时发现此类提示注入漏洞。



## **30. Who Speaks for the Trigger? Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers**

谁为触发器说话？后台混合专家变形器中的动态专家路由 cs.CR

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13462v1) [paper-pdf](http://arxiv.org/pdf/2510.13462v1)

**Authors**: Xin Zhao, Xiaojun Chen, Bingshan Liu, Haoyu Gao, Zhendong Zhao, Yilong Chen

**Abstract**: Large language models (LLMs) with Mixture-of-Experts (MoE) architectures achieve impressive performance and efficiency by dynamically routing inputs to specialized subnetworks, known as experts. However, this sparse routing mechanism inherently exhibits task preferences due to expert specialization, introducing a new and underexplored vulnerability to backdoor attacks. In this work, we investigate the feasibility and effectiveness of injecting backdoors into MoE-based LLMs by exploiting their inherent expert routing preferences. We thus propose BadSwitch, a novel backdoor framework that integrates task-coupled dynamic trigger optimization with a sensitivity-guided Top-S expert tracing mechanism. Our approach jointly optimizes trigger embeddings during pretraining while identifying S most sensitive experts, subsequently constraining the Top-K gating mechanism to these targeted experts. Unlike traditional backdoor attacks that rely on superficial data poisoning or model editing, BadSwitch primarily embeds malicious triggers into expert routing paths with strong task affinity, enabling precise and stealthy model manipulation. Through comprehensive evaluations across three prominent MoE architectures (Switch Transformer, QwenMoE, and DeepSeekMoE), we demonstrate that BadSwitch can efficiently hijack pre-trained models with up to 100% success rate (ASR) while maintaining the highest clean accuracy (ACC) among all baselines. Furthermore, BadSwitch exhibits strong resilience against both text-level and model-level defense mechanisms, achieving 94.07% ASR and 87.18% ACC on the AGNews dataset. Our analysis of expert activation patterns reveals fundamental insights into MoE vulnerabilities. We anticipate this work will expose security risks in MoE systems and contribute to advancing AI safety.

摘要: 具有专家混合（MoE）架构的大型语言模型（LLM）通过将输入动态路由到专门的子网络（称为专家）来实现令人印象深刻的性能和效率。然而，由于专家专业化，这种稀疏路由机制本质上会表现出任务偏好，从而引入了一个新的且未充分探索的后门攻击漏洞。在这项工作中，我们研究了通过利用基于教育部的LLM固有的专家路由偏好向其注入后门的可行性和有效性。因此，我们提出了BadSwitch，这是一种新型后门框架，它将任务耦合的动态触发优化与敏感性引导的Top-S专家跟踪机制集成在一起。我们的方法在预训练期间联合优化触发嵌入，同时识别S个最敏感的专家，随后将Top-K门控机制限制到这些目标专家。与依赖表面数据中毒或模型编辑的传统后门攻击不同，BadSwitch主要将恶意触发器嵌入到具有强任务亲和力的专家路由路径中，实现精确且隐蔽的模型操纵。通过对三种主要MoE架构（Switch Transformer、QwenMoE和DeepSeekMoE）的全面评估，我们证明BadSwitch可以有效劫持预训练模型，成功率高达100%（ZR），同时保持所有基线中最高的清理准确度（ACC）。此外，BadSwitch对文本级和模型级防御机制都表现出强大的弹性，在AGNews数据集中实现了94.07%的ASB和87.18%的ACC。我们对专家激活模式的分析揭示了对MoE漏洞的基本见解。我们预计这项工作将暴露MoE系统的安全风险，并有助于提高人工智能的安全性。



## **31. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

个人可以操纵多主体的集体决策吗？ cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2509.16494v2) [paper-pdf](http://arxiv.org/pdf/2509.16494v2)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

摘要: 个体大型语言模型（LLM）已在医疗保健和法律等各个领域展现出强大的能力。最近的研究还表明，协调的多智能体系统通过协作表现出增强的决策和推理能力。然而，由于单个LLM的脆弱性以及访问多代理系统中所有代理的困难，出现了一个关键问题：如果攻击者只知道一个代理，他们还能生成能够误导集体决策的对抗样本吗？为了探索这个问题，我们将其描述为一个信息不完整的游戏，其中攻击者只知道一个目标代理，并且缺乏对系统中其他代理的了解。通过这个公式，我们提出了M-Spoiler，这是一个模拟多智能体系统内的智能体交互以生成对抗样本的框架。然后使用这些样本来操纵目标系统中的目标代理，误导系统的协作决策过程。更具体地说，M-Spoiler引入了一种顽固代理，它通过模拟目标系统中代理的潜在顽固反应来积极帮助优化对抗样本。这增强了生成的对抗样本误导系统的有效性。通过针对各种任务的广泛实验，我们的研究结果证实了多代理系统中单个代理的知识所带来的风险，并证明了我们框架的有效性。我们还探索了几种防御机制，表明我们提出的攻击框架仍然比基线更有效，强调了进一步研究防御策略的必要性。



## **32. SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG**

SeCon-RAG：值得信赖的RAG的两阶段语义过滤和免预算框架 cs.CL

Accepted at NeurIPS 2025

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.09710v2) [paper-pdf](http://arxiv.org/pdf/2510.09710v2)

**Authors**: Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan, Yang Liu, Xiaojun Jia

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to corpus poisoning and contamination attacks, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. To address this problem, we propose a two-stage semantic filtering and conflict-free framework for trustworthy RAG. In the first stage, we perform a joint filter with semantic and cluster-based filtering which is guided by the Entity-intent-relation extractor (EIRE). EIRE extracts entities, latent objectives, and entity relations from both the user query and filtered documents, scores their semantic relevance, and selectively adds valuable documents into the clean retrieval database. In the second stage, we proposed an EIRE-guided conflict-aware filtering module, which analyzes semantic consistency between the query, candidate answers, and retrieved knowledge before final answer generation, filtering out internal and external contradictions that could mislead the model. Through this two-stage process, SeCon-RAG effectively preserves useful knowledge while mitigating conflict contamination, achieving significant improvements in both generation robustness and output trustworthiness. Extensive experiments across various LLMs and datasets demonstrate that the proposed SeCon-RAG markedly outperforms state-of-the-art defense methods.

摘要: 检索增强生成（RAG）系统利用外部知识增强大型语言模型（LLM），但容易受到语料库中毒和污染攻击，这可能会损害输出的完整性。现有的防御措施通常采用激进的过滤，导致不必要的有价值的信息丢失，并降低了生成的可靠性。为了解决这个问题，我们提出了一个两阶段的语义过滤和无冲突的框架值得信赖的RAG。在第一阶段中，我们执行一个联合过滤器与语义和基于聚类的过滤，这是指导的语义意图关系提取器（EIRE）。EERE从用户查询和过滤文档中提取实体、潜在目标和实体关系，对其语义相关性进行评分，并选择性地将有价值的文档添加到干净的检索数据库中。在第二阶段，我们提出了一个EIRE引导的冲突感知过滤模块，该模块在最终答案生成之前分析查询、候选答案和检索到的知识之间的语义一致性，过滤掉可能误导模型的内部和外部矛盾。通过这个两阶段过程，SeCon-RAG有效地保留了有用的知识，同时减轻了冲突污染，在发电稳健性和输出可信度方面实现了显着提高。跨各种LLM和数据集的广泛实验表明，拟议的SeCon-RAG明显优于最先进的防御方法。



## **33. SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs**

SHIELD：分类器引导的预算，实现更强大、更安全的LVLM cs.CL

Preprint

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13190v1) [paper-pdf](http://arxiv.org/pdf/2510.13190v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.

摘要: 大型视觉语言模型（LVLM）解锁了强大的多模式推理，但也扩大了攻击面，特别是通过在良性提示中隐藏有害目标的对抗性输入。我们提出SHIELD，这是一个轻量级的、模型不可知的预处理框架，它将细粒度的安全分类与特定类别的指导和显式动作（Block、Reframe、Forward）结合起来。与二元版主不同，SHIELD编写了量身定制的安全提示，无需再培训即可强制执行细致入微的拒绝或安全重定向。在五个基准和五个有代表性的LVLM中，SHIELD持续降低越狱和不跟随率，同时保持实用性。我们的方法是即插即用的，所产生的负担可以忽略不计，并且可以轻松扩展到新的攻击类型--作为弱对齐和强对齐LVLM的实用安全补丁。



## **34. RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs**

RAGE：适用于越狱LLM的参考感知和集成解码 cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.13901v1) [paper-pdf](http://arxiv.org/pdf/2510.13901v1)

**Authors**: Tuan T. Nguyen, John Le, Thai T. Vu, Willy Susilo, Heath Cooper

**Abstract**: Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.

摘要: 大型语言模型（LLM）在不同任务中取得了令人印象深刻的性能，但仍然容易受到绕过安全机制的越狱攻击。我们介绍了RAID-Aware（参考感知和集成解码），这是一个框架，通过制作对抗性后缀来系统性地探索这些弱点，这些后缀在保持流畅性的同时引入受限制的内容。RAID将离散令牌放松为连续嵌入，并通过联合目标对其进行优化，该目标（i）鼓励限制响应，（ii）合并反推感知规则化器以引导激活远离嵌入空间中的拒绝方向，以及（iii）应用一致性项来保持语义一致性和非冗余性。优化后，批评引导的解码过程通过平衡嵌入亲和力与语言模型可能性来将嵌入映射回令牌。这种集成产生的后缀既可以有效绕过防御，而且形式自然。在多个开源LLM上的实验表明，与最近的白盒和黑盒基线相比，与更少的查询和更低的计算成本实现了更高的攻击成功率。这些发现凸显了嵌入空间规范化对于理解和缓解LLM越狱漏洞的重要性。



## **35. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07505v2) [paper-pdf](http://arxiv.org/pdf/2510.07505v2)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然兼容各种MAS体系结构，我们的基准集中在规划者-执行器结构，这是一个实用的和广泛采用的设计。通过大量的实验，我们发现：（1）弱规划器比弱执行器更严重地降低了清洁任务的整体性能;（2）虽然规划器的内存模块是必不可少的，但执行器的内存模块并不影响清洁任务的性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击在误导系统方面特别有效。这些发现提供了可操作的见解，提高MAS的鲁棒性，并奠定了基础，在多智能体设置的原则性防御。



## **36. Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection**

守护护栏：分类学驱动的越狱检测方法 cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.13893v1) [paper-pdf](http://arxiv.org/pdf/2510.13893v1)

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards.

摘要: 越狱技术对大型语言模型（LLM）的安全构成了重大威胁。现有的防御通常专注于单轮攻击，缺乏跨语言的覆盖，并且依赖于有限的分类法，这些分类法要么无法捕捉攻击策略的全部多样性，要么强调风险类别而不是越狱技术。为了加深对越狱技术有效性的了解，我们进行了一次结构化的红色团队挑战。我们实验的结果是多方面的。首先，我们开发了50种越狱策略的全面分层分类，将之前的分类整合并扩展到七个大系列，包括模仿、说服、特权升级、认知过载、混淆、目标冲突和数据中毒。其次，我们分析了从挑战中收集的数据，以检查不同攻击类型的流行率和成功率，从而深入了解特定越狱策略如何利用模型漏洞并导致失调。第三，我们对流行的LLM进行越狱检测进行基准测试，评估分类学引导的提示对改进自动检测的好处。最后，我们编制了一个新的意大利数据集，包含1364个多回合对抗对话，并使用我们的分类法进行了注释，从而能够研究对抗意图逐渐出现并成功绕过传统保障措施的相互作用。



## **37. IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion**

通过令牌对比表示增强和多粒度融合进行IP增强多模式恶意URL检测 cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12395v1) [paper-pdf](http://arxiv.org/pdf/2510.12395v1)

**Authors**: Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

**Abstract**: Malicious URL detection remains a critical cybersecurity challenge as adversaries increasingly employ sophisticated evasion techniques including obfuscation, character-level perturbations, and adversarial attacks. Although pre-trained language models (PLMs) like BERT have shown potential for URL analysis tasks, three limitations persist in current implementations: (1) inability to effectively model the non-natural hierarchical structure of URLs, (2) insufficient sensitivity to character-level obfuscation, and (3) lack of mechanisms to incorporate auxiliary network-level signals such as IP addresses-all essential for robust detection. To address these challenges, we propose CURL-IP, an advanced multi-modal detection framework incorporating three key innovations: (1) Token-Contrastive Representation Enhancer, which enhances subword token representations through token-aware contrastive learning to produce more discriminative and isotropic embeddings; (2) Cross-Layer Multi-Scale Aggregator, employing hierarchical aggregation of Transformer outputs via convolutional operations and gated MLPs to capture both local and global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler that decomposes URL-IP features into localized block units and computes cross-modal attention weights at the block level, enabling fine-grained inter-modal interaction. This architecture enables simultaneous preservation of fine-grained lexical cues, contextual semantics, and integration of network-level signals. Our evaluation on large-scale real-world datasets shows the framework significantly outperforms state-of-the-art baselines across binary and multi-class classification tasks.

摘要: 恶意URL检测仍然是一个关键的网络安全挑战，因为对手越来越多地使用复杂的规避技术，包括混淆、字符级扰动和对抗性攻击。尽管BERT等预训练语言模型（PLM）已显示出URL分析任务的潜力，但当前的实现中仍然存在三个局限性：（1）无法有效地建模URL的非自然分层结构，（2）对字符级混淆的敏感性不足，（3）缺乏纳入辅助网络级信号（例如IP地址）的机制--所有这些对于鲁棒检测来说都至关重要。为了应对这些挑战，我们提出了CROL-IP，这是一种先进的多模式检测框架，融合了三项关键创新：（1）令牌对比表示增强器，它通过令牌感知的对比学习来增强子词令牌表示，以产生更具区分性和各向同性的嵌入;（2）跨层多尺度聚合器，通过卷积运算和门控MLP采用Transformer输出的分层聚合来跨层捕获局部和全局语义模式;和（3）绑定多模式耦合器，将URL-IP特征分解为局部块单元，并计算块级别的跨模式注意力权重，从而实现细粒度的模式间交互。该架构能够同时保存细粒度的词汇线索、上下文语义和网络级信号的集成。我们对大规模现实世界数据集的评估表明，该框架在二元和多类分类任务中的表现显着优于最先进的基线。



## **38. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

L2 M-AID：通过融合大型语言模型的语义推理与多智能体强化学习来自主网络物理防御（预印本） cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07363v2) [paper-pdf](http://arxiv.org/pdf/2510.07363v2)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

摘要: 工业物联网（IIoT）的日益集成使关键的网络物理系统面临复杂的多阶段攻击，这些攻击无法逃避缺乏上下文感知的传统防御。本文介绍了L2 M-AID，这是一种新型的自主工业防御框架，使用LLM授权的多智能体强化学习。L2 M-AID组织了一个协作代理团队，每个代理都由大型语言模型（LLM）驱动，以实现自适应和弹性的安全性。核心创新在于两种人工智能范式的深度融合：我们利用LLM作为语义桥梁，将庞大的非结构化遥感数据转化为丰富的上下文状态表示，使代理能够推理对手意图，而不仅仅是匹配模式。这种语义感知状态使多智能体强化学习（MARL）算法MAPPO能够学习复杂的合作策略。MARL奖励功能经过独特设计，旨在平衡安全目标（威胁消除）与运营必要性，明确惩罚破坏物理过程稳定性的行为。为了验证我们的方法，我们对基准SWaT数据集和基于MITRE ATA & CK for ICS框架生成的新型合成数据集进行了广泛的实验。结果表明，L2 M-AID在关键指标上的表现显着优于传统IDS、深度学习异常检测器和单代理RL基线，实现了97.2%的检测率，同时将误报率降低了80%以上，并将响应时间提高了四倍。至关重要的是，它在维持物理过程稳定性方面表现出色，为保护关键国家基础设施提供了强大的新范式。



## **39. Cross-Modal Safety Alignment: Is textual unlearning all you need?**

跨模态安全对齐：你需要的只是文本遗忘吗？ cs.CL

Accepted by EMNLP 2024 Findings

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2406.02575v2) [paper-pdf](http://arxiv.org/pdf/2406.02575v2)

**Authors**: Trishna Chakraborty, Erfan Shayegani, Zikui Cai, Nael Abu-Ghazaleh, M. Salman Asif, Yue Dong, Amit K. Roy-Chowdhury, Chengyu Song

**Abstract**: Recent studies reveal that integrating new modalities into Large Language Models (LLMs), such as Vision-Language Models (VLMs), creates a new attack surface that bypasses existing safety training techniques like Supervised Fine-tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). While further SFT and RLHF-based safety training can be conducted in multi-modal settings, collecting multi-modal training datasets poses a significant challenge. Inspired by the structural design of recent multi-modal models, where, regardless of the combination of input modalities, all inputs are ultimately fused into the language space, we aim to explore whether unlearning solely in the textual domain can be effective for cross-modality safety alignment. Our evaluation across six datasets empirically demonstrates the transferability -- textual unlearning in VLMs significantly reduces the Attack Success Rate (ASR) to less than 8\% and in some cases, even as low as nearly 2\% for both text-based and vision-text-based attacks, alongside preserving the utility. Moreover, our experiments show that unlearning with a multi-modal dataset offers no potential benefits but incurs significantly increased computational demands, possibly up to 6 times higher.

摘要: 最近的研究表明，将新的模式集成到大型语言模型（LLM）中，例如视觉语言模型（VLM），可以创建一个新的攻击表面，绕过现有的安全训练技术，例如监督微调（SFT）和带人类反馈的强化学习（RL HF）。虽然进一步的基于SFT和WLHF的安全培训可以在多模式环境中进行，但收集多模式训练数据集构成了重大挑战。受最近多模式模型的结构设计的启发，无论输入模式的组合如何，所有输入最终都会融合到语言空间中，我们的目标是探索仅在文本领域放弃学习是否可以有效地实现跨模式的安全对齐。我们对六个数据集的评估从经验上证明了可移植性--VLM中的文本取消学习将攻击成功率（ASB）显着降低到8%以下，在某些情况下，对于基于文本和基于视觉文本的攻击，攻击成功率甚至低至近2%，同时保留了实用性。此外，我们的实验表明，放弃使用多模式数据集的学习不会带来潜在的好处，但会导致计算需求显着增加，可能高达6倍。



## **40. Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs**

揭开Graph-LLM的漏洞：对TAG的可解释多维对抗攻击 cs.LG

12 pages, 4 figures

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12233v1) [paper-pdf](http://arxiv.org/pdf/2510.12233v1)

**Authors**: Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

**Abstract**: Graph Neural Networks (GNNs) have become a pivotal framework for modeling graph-structured data, enabling a wide range of applications from social network analysis to molecular chemistry. By integrating large language models (LLMs), text-attributed graphs (TAGs) enhance node representations with rich textual semantics, significantly boosting the expressive power of graph-based learning. However, this sophisticated synergy introduces critical vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both their structural topology and textual attributes. Although specialized attack methods have been designed for each of these aspects, no work has yet unified them into a comprehensive approach. In this work, we propose the Interpretable Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial attack framework designed to orchestrate multi-level perturbations across both graph structure and textual features. IMDGA utilizes three tightly integrated modules to craft attacks that balance interpretability and impact, enabling a deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical analysis and comprehensive empirical evaluations on diverse datasets and architectures, IMDGA demonstrates superior interpretability, attack effectiveness, stealthiness, and robustness compared to existing methods. By exposing critical weaknesses in TAG representation learning, this work uncovers a previously underexplored semantic dimension of vulnerability in Graph-LLMs, offering valuable insights for improving their resilience. Our code and resources are publicly available at https://anonymous.4open.science/r/IMDGA-7289.

摘要: 图神经网络（GNN）已经成为对图结构数据进行建模的关键框架，能够实现从社交网络分析到分子化学的广泛应用。通过集成大型语言模型（LLM），文本属性图（TAG）增强了具有丰富文本语义的节点表示，显着提高了基于图的学习的表达能力。然而，这种复杂的协同作用引入了关键的漏洞，因为Graph-LLM容易受到对其结构拓扑和文本属性的对抗性攻击。虽然专门的攻击方法已被设计用于这些方面的每一个，还没有工作将它们统一成一个全面的方法。在这项工作中，我们提出了可解释多维图攻击（IMDGA），这是一种新型的以人为中心的对抗攻击框架，旨在协调跨图结构和文本特征的多层扰动。IMDGA利用三个紧密集成的模块来设计平衡可解释性和影响的攻击，从而能够更深入地了解Graph-LLM漏洞。通过对不同数据集和架构进行严格的理论分析和全面的实证评估，IMDGA展示了与现有方法相比更出色的可解释性、攻击有效性、隐蔽性和鲁棒性。通过揭露TAG表示学习中的关键弱点，这项工作揭示了Graph-LLM中先前未充分探索的漏洞语义维度，为提高其弹性提供了宝贵的见解。我们的代码和资源可在https://anonymous.4open.science/r/IMDGA-7289上公开获取。



## **41. HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities**

HackWorld：评估计算机使用代理利用Web应用程序漏洞的能力 cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12200v1) [paper-pdf](http://arxiv.org/pdf/2510.12200v1)

**Authors**: Xiaoxue Ren, Penghao Jiang, Kaixin Li, Zhiyong Huang, Xiaoning Du, Jiaojiao Jiang, Zhenchang Xing, Jiamou Sun, Terry Yue Zhuo

**Abstract**: Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input handling. Using a Capture-the-Flag (CTF) setup, it tests CUAs' capacity to identify and exploit these weaknesses while navigating complex web interfaces. Evaluation of state-of-the-art CUAs reveals concerning trends: exploitation rates below 12% and low cybersecurity awareness. CUAs often fail at multi-step attack planning and misuse security tools. These results expose the current limitations of CUAs in web security contexts and highlight opportunities for developing more security-aware agents capable of effective vulnerability detection and exploitation.

摘要: Web应用程序是网络攻击的主要目标，是关键服务和敏感数据的网关。传统的渗透测试成本高昂，需要大量的专业知识，因此很难随着不断增长的网络生态系统而扩展。虽然语言模型代理在网络安全方面表现出了希望，但现代Web应用程序需要视觉理解，动态内容处理和多步交互，只有计算机使用代理（CUA）才能执行。然而，它们通过图形界面发现和利用漏洞的能力在很大程度上仍未得到开发。我们展示了HackWorld，这是第一个用于系统评估CUA通过视觉交互利用Web应用程序漏洞的能力的框架。与经过清理的基准测试不同，HackWorld包括36个现实世界的应用程序，涵盖11个框架和7种语言，具有注入漏洞、身份验证绕过和不安全的输入处理等现实缺陷。它使用Capture-the-Flag（CTF）设置来测试CUA在导航复杂Web界面时识别和利用这些弱点的能力。对最先进的CUA的评估揭示了令人担忧的趋势：利用率低于12%，网络安全意识较低。CUA经常在多步骤攻击计划方面失败并滥用安全工具。这些结果暴露了CUA当前在网络安全环境中的局限性，并强调了开发能够有效检测和利用漏洞的更具安全意识的代理的机会。



## **42. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

当推理中的“能力”打开脆弱之门：通过新颖复杂密码越狱LLM cs.CL

Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2402.10601v5) [paper-pdf](http://arxiv.org/pdf/2402.10601v5)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Md Nayem Uddin, Aswin RRV, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.

摘要: 大型语言模型（LLM）安全性的最新进展主要集中在减轻用自然语言或常用密码（例如Base 64）精心设计的攻击，这些攻击可能会集成到较新模型的安全培训中。然而，我们揭示了一个自相矛盾的漏洞：随着LLM在推理方面的进步，它们无意中变得更容易受到新颖的越狱攻击。增强的推理使LLM能够解释复杂的指令并解码复杂的用户定义的密码，从而创造了可利用的安全漏洞。为了研究此漏洞，我们引入了使用自定义加密（ACE）的攻击，这是一种越狱技术，使用新颖的密码对恶意查询进行编码。扩展ACE，我们引入了使用自定义加密（LACE）的分层攻击，该加密应用多层密码来放大攻击复杂性。此外，我们还开发了CipherBench，这是一个旨在评估LLM解码加密良性文本的准确性的基准。我们的实验揭示了一个关键的权衡：解码密码能力更强的LLM更容易受到LACE的影响，gtt-oss-20 b的成功率从ACE下的60%上升到LACE下的72%。这些发现凸显了一个关键的见解：随着LLM越来越善于破译复杂的用户密码（其中许多密码无法预先包含在安全培训中），它们变得越来越容易被利用。



## **43. Attention-Aware GNN-based Input Defense against Multi-Turn LLM Jailbreak**

注意力意识的基于GNN的输入防御针对多回合LLM越狱 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2507.07146v2) [paper-pdf](http://arxiv.org/pdf/2507.07146v2)

**Authors**: Zixuan Huang, Kecheng Huang, Lihao Yin, Bowei He, Huiling Zhen, Mingxuan Yuan, Zili Shao

**Abstract**: Large Language Models (LLMs) have gained significant traction in various applications, yet their capabilities present risks for both constructive and malicious exploitation. Despite extensive training and fine-tuning efforts aimed at enhancing safety, LLMs remain susceptible to jailbreak attacks. Recently, the emergence of multi-turn attacks has intensified this vulnerability. Unlike single-turn attacks, multi-turn attacks incrementally escalate dialogue complexity, rendering them more challenging to detect and mitigate.   In this study, we introduce G-Guard, an innovative attention-aware Graph Neural Network (GNN)-based input classifier specifically designed to defend against multi-turn jailbreak attacks targeting LLMs. G-Guard constructs an entity graph for multi-turn queries, which captures the interrelationships between queries and harmful keywords that present in multi-turn queries. Furthermore, we propose an attention-aware augmentation mechanism that retrieves the most relevant single-turn query based on the ongoing multi-turn conversation. The retrieved query is incorporated as a labeled node within the graph, thereby enhancing the GNN's capacity to classify the current query as harmful or benign. Evaluation results show that G-Guard consistently outperforms all baselines across diverse datasets and evaluation metrics, demonstrating its efficacy as a robust defense mechanism against multi-turn jailbreak attacks.

摘要: 大型语言模型（LLM）在各种应用程序中获得了巨大的吸引力，但它们的功能存在建设性和恶意利用的风险。尽管针对提高安全性进行了广泛的培训和微调，但LLMs仍然容易受到越狱攻击。最近，多回合攻击的出现加剧了这种脆弱性。与单回合攻击不同，多回合攻击逐渐增加了对话的复杂性，使其更难以检测和缓解。   在这项研究中，我们介绍了G-Guard，这是一种创新的基于注意力感知的图神经网络（GNN）的输入分类器，专门用于防御针对LLM的多回合越狱攻击。G-Guard为多轮查询构建了一个实体图，该实体图捕捉了多轮查询中存在的查询和有害关键字之间的相互关系。此外，我们提出了一个注意力感知增强机制，检索最相关的单轮查询的基础上正在进行的多轮对话。检索到的查询被合并为图中的标记节点，从而增强了GNN将当前查询分类为有害或良性的能力。评估结果显示，G-Guard在不同数据集和评估指标中的表现始终优于所有基线，证明了其作为针对多回合越狱攻击的强大防御机制的有效性。



## **44. SafeMT: Multi-turn Safety for Multimodal Language Models**

SafeMT：多模式语言模型的多轮安全性 cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12133v1) [paper-pdf](http://arxiv.org/pdf/2510.12133v1)

**Authors**: Han Zhu, Juntao Dai, Jiaming Ji, Haoran Li, Chengkun Cai, Pengcheng Wen, Chi-Min Chan, Boyuan Chen, Yaodong Yang, Sirui Han, Yike Guo

**Abstract**: With the widespread use of multi-modal Large Language models (MLLMs), safety issues have become a growing concern. Multi-turn dialogues, which are more common in everyday interactions, pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation. To encourage the community to focus on the safety issues of these models in multi-turn dialogues, we introduce SafeMT, a benchmark that features dialogues of varying lengths generated from harmful queries accompanied by images. This benchmark consists of 10,000 samples in total, encompassing 17 different scenarios and four jailbreak methods. Additionally, we propose Safety Index (SI) to evaluate the general safety of MLLMs during conversations. We assess the safety of 17 models using this benchmark and discover that the risk of successful attacks on these models increases as the number of turns in harmful dialogues rises. This observation indicates that the safety mechanisms of these models are inadequate for recognizing the hazard in dialogue interactions. We propose a dialogue safety moderator capable of detecting malicious intent concealed within conversations and providing MLLMs with relevant safety policies. Experimental results from several open-source models indicate that this moderator is more effective in reducing multi-turn ASR compared to existed guard models.

摘要: 随着多模式大型语言模型（MLLM）的广泛使用，安全问题已成为人们日益关注的问题。多轮对话在日常互动中更常见，比单个提示带来的风险更大;然而，现有的基准没有充分考虑这种情况。为了鼓励社区在多轮对话中关注这些模型的安全问题，我们引入了SafeMT，这是一个基准，其特点是由伴随图像的有害查询生成的不同长度的对话。该基准测试总共包含10，000个样本，涵盖17种不同的场景和4种越狱方法。此外，我们还提出了安全指数（SI）来评估MLLM在对话期间的总体安全性。我们使用该基准评估了17个模型的安全性，并发现随着有害对话轮数的增加，对这些模型进行成功攻击的风险也会增加。这一观察表明，这些模型的安全机制不足以识别对话互动中的危险。我们提出了一个对话安全版主，能够检测隐藏在对话中的恶意意图，并为MLLM提供相关安全政策。多个开源模型的实验结果表明，与现有的警卫模型相比，这种调节器在减少多圈ASO方面更有效。



## **45. GraphRAG under Fire**

GraphRAG受到攻击 cs.LG

13 pages. Accepted by IEEE Symposium on Security and Privacy 2026  (S&P 2026)

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2501.14050v4) [paper-pdf](http://arxiv.org/pdf/2501.14050v4)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: existing RAG poisoning attacks are less effective under GraphRAG than conventional RAG, due to GraphRAG's graph-based indexing and retrieval; yet, the same features also create new attack surfaces. We present GragPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GragPoison employs three key strategies: (i) relation injection to introduce false knowledge, (ii) relation enhancement to amplify poisoning influence, and (iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GragPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text) on multiple variations of GraphRAG. We also explore potential defensive measures and their limitations, identifying promising directions for future research.

摘要: GraphRAG通过将外部知识结构化为多尺度知识图，使语言模型能够在生成中集成广泛的上下文和粒度细节，从而推进了检索增强生成（RAG）。虽然GraphRAG在各个领域都取得了成功，但其安全影响在很大程度上仍未被探索。为了弥合这一差距，这项工作研究了GraphRAG对中毒攻击的脆弱性，揭示了一个有趣的安全悖论：由于GraphRAG的基于图形的索引和检索，现有的RAG中毒攻击在GraphRAG下不如传统RAG有效;然而，相同的功能也创建了新的攻击面。我们提出了GragPoison，这是一种新颖的攻击，它利用底层知识图中的共享关系来制作能够同时破坏多个查询的中毒文本。GragPoison采用三种关键策略：（i）关系注入以引入虚假知识，（ii）关系增强以放大中毒影响，以及（iii）叙事生成以将恶意内容嵌入连贯文本中。对不同数据集和模型的经验评估表明，GragPoison在对GraphRAG的多种变体的有效性（高达98%的成功率）和可扩展性（使用不到68%的中毒文本）方面大大优于现有的攻击。我们还探索潜在的防御措施及其局限性，为未来研究确定有希望的方向。



## **46. Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing**

使用高级文本预处理对传统、LLM生成和对抗性网络钓鱼电子邮件进行稳健的基于ML的检测 cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11915v1) [paper-pdf](http://arxiv.org/pdf/2510.11915v1)

**Authors**: Deeksha Hareesha Kulal, Chidozie Princewill Arannonu, Afsah Anwar, Nidhi Rastogi, Quamar Niyaz

**Abstract**: Phishing remains a critical cybersecurity threat, especially with the advent of large language models (LLMs) capable of generating highly convincing malicious content. Unlike earlier phishing attempts which are identifiable by grammatical errors, misspellings, incorrect phrasing, and inconsistent formatting, LLM generated emails are grammatically sound, contextually relevant, and linguistically natural. These advancements make phishing emails increasingly difficult to distinguish from legitimate ones, challenging traditional detection mechanisms. Conventional phishing detection systems often fail when faced with emails crafted by LLMs or manipulated using adversarial perturbation techniques. To address this challenge, we propose a robust phishing email detection system featuring an enhanced text preprocessing pipeline. This pipeline includes spelling correction and word splitting to counteract adversarial modifications and improve detection accuracy. Our approach integrates widely adopted natural language processing (NLP) feature extraction techniques and machine learning algorithms. We evaluate our models on publicly available datasets comprising both phishing and legitimate emails, achieving a detection accuracy of 94.26% and F1-score of 84.39% in model deployment setting. To assess robustness, we further evaluate our models using adversarial phishing samples generated by four attack methods in Python TextAttack framework. Additionally, we evaluate models' performance against phishing emails generated by LLMs including ChatGPT and Llama. Results highlight the resilience of models against evolving AI-powered phishing threats.

摘要: 网络钓鱼仍然是一个严重的网络安全威胁，特别是随着能够生成高度令人信服的恶意内容的大型语言模型（LLM）的出现。与早期的网络钓鱼尝试（可通过语法错误、拼写错误、措辞不正确和格式不一致）不同，LLM生成的电子邮件语法健全、上下文相关且语言自然。这些进步使得网络钓鱼电子邮件越来越难以与合法电子邮件区分开来，从而挑战了传统的检测机制。当面对由LLM制作或使用对抗性干扰技术操纵的电子邮件时，传统的网络钓鱼检测系统通常会失败。为了应对这一挑战，我们提出了一种强大的网络钓鱼电子邮件检测系统，具有增强的文本预处理管道。该管道包括拼写纠正和单词拆分，以抵消对抗性修改并提高检测准确性。我们的方法集成了广泛采用的自然语言处理（NLP）特征提取技术和机器学习算法。我们在包括网络钓鱼和合法电子邮件的公开数据集上评估了我们的模型，在模型部署设置中实现了94.26%的检测准确率和84.39%的F1评分。为了评估稳健性，我们使用Python文本攻击框架中四种攻击方法生成的对抗性网络钓鱼样本进一步评估我们的模型。此外，我们还评估模型针对ChatGPT和Llama等LLM生成的网络钓鱼电子邮件的性能。结果凸显了模型对不断变化的人工智能驱动网络钓鱼威胁的弹性。



## **47. Countermind: A Multi-Layered Security Architecture for Large Language Models**

Countermind：大型语言模型的多层安全架构 cs.CR

33 pages, 3 figures, 6 tables. Keywords: LLM security;  defense-in-depth; prompt injection; activation steering; multimodal sandbox;  threat modeling

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11837v1) [paper-pdf](http://arxiv.org/pdf/2510.11837v1)

**Authors**: Dominik Schwarz

**Abstract**: The security of Large Language Model (LLM) applications is fundamentally challenged by "form-first" attacks like prompt injection and jailbreaking, where malicious instructions are embedded within user inputs. Conventional defenses, which rely on post hoc output filtering, are often brittle and fail to address the root cause: the model's inability to distinguish trusted instructions from untrusted data. This paper proposes Countermind, a multi-layered security architecture intended to shift defenses from a reactive, post hoc posture to a proactive, pre-inference, and intra-inference enforcement model. The architecture proposes a fortified perimeter designed to structurally validate and transform all inputs, and an internal governance mechanism intended to constrain the model's semantic processing pathways before an output is generated. The primary contributions of this work are conceptual designs for: (1) A Semantic Boundary Logic (SBL) with a mandatory, time-coupled Text Crypter intended to reduce the plaintext prompt injection attack surface, provided all ingestion paths are enforced. (2) A Parameter-Space Restriction (PSR) mechanism, leveraging principles from representation engineering, to dynamically control the LLM's access to internal semantic clusters, with the goal of mitigating semantic drift and dangerous emergent behaviors. (3) A Secure, Self-Regulating Core that uses an OODA loop and a learning security module to adapt its defenses based on an immutable audit log. (4) A Multimodal Input Sandbox and Context-Defense mechanisms to address threats from non-textual data and long-term semantic poisoning. This paper outlines an evaluation plan designed to quantify the proposed architecture's effectiveness in reducing the Attack Success Rate (ASR) for form-first attacks and to measure its potential latency overhead.

摘要: 大型语言模型（LLM）应用程序的安全性从根本上受到提示注入和越狱等“形式优先”攻击的挑战，其中恶意指令嵌入在用户输入中。依赖于事后输出过滤的传统防御通常很脆弱，无法解决根本原因：模型无法区分可信指令与不可信数据。本文提出了Counterend，这是一种多层安全架构，旨在将防御从反应性、事后姿态转变为主动性、预推理和内推理实施模型。该架构提出了一个旨在从结构上验证和转换所有输入的强化边界，以及一个旨在在生成输出之前限制模型的语义处理路径的内部治理机制。这项工作的主要贡献是以下方面的概念设计：（1）具有强制性、时间耦合的文本加密器的语义边界逻辑（SBL），旨在减少明文提示注入攻击表面，前提是强制执行所有摄入路径。(2)参数空间限制（PPC）机制，利用表示工程的原则，动态控制LLM对内部语义集群的访问，目标是减轻语义漂移和危险的紧急行为。(3)一个安全、自我调节的核心，使用OODA循环和学习安全模块来根据不可变的审计日志调整其防御。(4)多模式输入沙盒和上下文防御机制，可解决来自非文本数据和长期语义中毒的威胁。本文概述了一个评估计划，旨在量化拟议架构在降低形式优先攻击的攻击成功率（ASB）方面的有效性，并衡量其潜在的延迟负担。



## **48. LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings**

LLMAtKGE：大型语言模型作为知识图嵌入的可解释攻击者 cs.CL

13 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11584v1) [paper-pdf](http://arxiv.org/pdf/2510.11584v1)

**Authors**: Ting Li, Yang Yang, Yipeng Yu, Liang Yao, Guoqing Chao, Ruifeng Xu

**Abstract**: Adversarial attacks on knowledge graph embeddings (KGE) aim to disrupt the model's ability of link prediction by removing or inserting triples. A recent black-box method has attempted to incorporate textual and structural information to enhance attack performance. However, it is unable to generate human-readable explanations, and exhibits poor generalizability. In the past few years, large language models (LLMs) have demonstrated powerful capabilities in text comprehension, generation, and reasoning. In this paper, we propose LLMAtKGE, a novel LLM-based framework that selects attack targets and generates human-readable explanations. To provide the LLM with sufficient factual context under limited input constraints, we design a structured prompting scheme that explicitly formulates the attack as multiple-choice questions while incorporating KG factual evidence. To address the context-window limitation and hesitation issues, we introduce semantics-based and centrality-based filters, which compress the candidate set while preserving high recall of attack-relevant information. Furthermore, to efficiently integrate both semantic and structural information into the filter, we precompute high-order adjacency and fine-tune the LLM with a triple classification task to enhance filtering performance. Experiments on two widely used knowledge graph datasets demonstrate that our attack outperforms the strongest black-box baselines and provides explanations via reasoning, and showing competitive performance compared with white-box methods. Comprehensive ablation and case studies further validate its capability to generate explanations.

摘要: 对知识图嵌入（KGE）的对抗攻击旨在通过删除或插入三重组来破坏模型的链接预测能力。最近的一种黑匣子方法试图合并文本和结构信息以增强攻击性能。然而，它无法生成人类可读的解释，并且表现出较差的概括性。在过去的几年里，大型语言模型（LLM）在文本理解、生成和推理方面展示了强大的能力。在本文中，我们提出了LLMAtKGE，这是一个基于LLM的新型框架，可以选择攻击目标并生成人类可读的解释。为了在有限的输入限制下为LLM提供足够的事实背景，我们设计了一个结构化的提示方案，该方案将攻击明确地制定为多项选择题，同时纳入KG事实证据。为了解决上下文窗口限制和犹豫问题，我们引入了基于语义和基于中心性的过滤器，它们压缩候选集，同时保留攻击相关信息的高召回率。此外，为了有效地将语义和结构信息集成到过滤器中，我们预先计算了高位邻近并通过三重分类任务微调LLM，以增强过滤性能。在两个广泛使用的知识图谱数据集上的实验表明，我们的攻击优于最强的黑盒基线，并通过推理提供解释，与白盒方法相比，表现出有竞争力的性能。全面的消融和病例研究进一步验证了其产生解释的能力。



## **49. The Enemy from Within: A Study of Political Delegitimization Discourse in Israeli Political Speech**

来自内部的敌人：以色列政治演讲中政治去合法化话语的研究 cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2508.15524v2) [paper-pdf](http://arxiv.org/pdf/2508.15524v2)

**Authors**: Naama Rivlin-Angert, Guy Mor-Lan

**Abstract**: We present the first large-scale computational study of political delegitimization discourse (PDD), defined as symbolic attacks on the normative validity of political entities. We curate and manually annotate a novel Hebrew-language corpus of 10,410 sentences drawn from Knesset speeches (1993-2023), Facebook posts (2018-2021), and leading news outlets, of which 1,812 instances (17.4\%) exhibit PDD and 642 carry additional annotations for intensity, incivility, target type, and affective framing. We introduce a two-stage classification pipeline combining finetuned encoder models and decoder LLMs. Our best model (DictaLM 2.0) attains an F$_1$ of 0.74 for binary PDD detection and a macro-F$_1$ of 0.67 for classification of delegitimization characteristics. Applying this classifier to longitudinal and cross-platform data, we see a marked rise in PDD over three decades, higher prevalence on social media versus parliamentary debate, greater use by male than female politicians, and stronger tendencies among right-leaning actors - with pronounced spikes during election campaigns and major political events. Our findings demonstrate the feasibility and value of automated PDD analysis for understanding democratic discourse.

摘要: 我们对政治去合法性话语（PDD）进行了首次大规模计算研究，PDD被定义为对政治实体规范有效性的象征性攻击。我们策划并手动注释了一个新颖的希伯来语数据库，其中包含10，410个句子，取自以色列议会演讲（1993-2023年）、Facebook帖子（2018-2021年）和领先新闻媒体，其中1，812个实例（17.4%）表现出PDD，642个实例带有强度、礼貌、目标类型和情感框架的额外注释。我们引入了一个两阶段分类流水线，结合了微调编码器模型和解码器LLM。我们的最佳模型（DictaLM 2.0）对于二进制PDD检测，F$_1$为0.74，对于去合法化特征分类，宏F$_1 $为0.67。将这种分类器应用于纵向和跨平台数据，我们看到三十年来PDD显着上升，社交媒体上的流行率高于议会辩论，男性政客的使用率高于女性政客，右倾行为者的倾向更强--在竞选和重大政治活动期间出现明显峰值。我们的研究结果证明了自动PDD分析对于理解民主话语的可行性和价值。



## **50. Large Language Models Are Effective Code Watermarkers**

大型语言模型是有效的代码水印 cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11251v1) [paper-pdf](http://arxiv.org/pdf/2510.11251v1)

**Authors**: Rui Xu, Jiawei Chen, Zhaoxia Yin, Cong Kong, Xinpeng Zhang

**Abstract**: The widespread use of large language models (LLMs) and open-source code has raised ethical and security concerns regarding the distribution and attribution of source code, including unauthorized redistribution, license violations, and misuse of code for malicious purposes. Watermarking has emerged as a promising solution for source attribution, but existing techniques rely heavily on hand-crafted transformation rules, abstract syntax tree (AST) manipulation, or task-specific training, limiting their scalability and generality across languages. Moreover, their robustness against attacks remains limited. To address these limitations, we propose CodeMark-LLM, an LLM-driven watermarking framework that embeds watermark into source code without compromising its semantics or readability. CodeMark-LLM consists of two core components: (i) Semantically Consistent Embedding module that applies functionality-preserving transformations to encode watermark bits, and (ii) Differential Comparison Extraction module that identifies the applied transformations by comparing the original and watermarked code. Leveraging the cross-lingual generalization ability of LLM, CodeMark-LLM avoids language-specific engineering and training pipelines. Extensive experiments across diverse programming languages and attack scenarios demonstrate its robustness, effectiveness, and scalability.

摘要: 大型语言模型（LLM）和开源代码的广泛使用引发了有关源代码分发和归属的道德和安全问题，包括未经授权的重新分发、违反许可证以及出于恶意目的滥用代码。水印已成为源属性的一种有希望的解决方案，但现有技术严重依赖手工制作的转换规则、抽象语法树（AST）操作或特定任务的训练，限制了它们跨语言的可扩展性和通用性。此外，它们对攻击的稳健性仍然有限。为了解决这些限制，我们提出了CodeMark-LLM，这是一个LLM驱动的水印框架，它将水印嵌入到源代码中，而不会损害其语义或可读性。CodeMark-LLM由两个核心组件组成：（i）语义一致嵌入模块，应用功能保留变换来编码水印位，以及（ii）差异比较提取模块，通过比较原始代码和带水印代码来识别应用的变换。利用LLM的跨语言概括能力，CodeMark-LLM避免了特定于语言的工程和培训管道。跨不同编程语言和攻击场景的广泛实验证明了其稳健性、有效性和可扩展性。



