# Latest Adversarial Attack Papers
**update at 2025-12-01 09:06:56**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attention-Guided Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Models**

视觉-语言-动作模型的注意力引导补丁式稀疏对抗攻击 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21663v1) [paper-pdf](https://arxiv.org/pdf/2511.21663v1)

**Authors**: Naifu Zhang, Wei Tao, Xi Xiao, Qianpu Sun, Yuxin Zheng, Wentao Mo, Peiqiang Wang, Nan Zhang

**Abstract**: In recent years, Vision-Language-Action (VLA) models in embodied intelligence have developed rapidly. However, existing adversarial attack methods require costly end-to-end training and often generate noticeable perturbation patches. To address these limitations, we propose ADVLA, a framework that directly applies adversarial perturbations on features projected from the visual encoder into the textual feature space. ADVLA efficiently disrupts downstream action predictions under low-amplitude constraints, and attention guidance allows the perturbations to be both focused and sparse. We introduce three strategies that enhance sensitivity, enforce sparsity, and concentrate perturbations. Experiments demonstrate that under an $L_{\infty}=4/255$ constraint, ADVLA combined with Top-K masking modifies less than 10% of the patches while achieving an attack success rate of nearly 100%. The perturbations are concentrated on critical regions, remain almost imperceptible in the overall image, and a single-step iteration takes only about 0.06 seconds, significantly outperforming conventional patch-based attacks. In summary, ADVLA effectively weakens downstream action predictions of VLA models under low-amplitude and locally sparse conditions, avoiding the high training costs and conspicuous perturbations of traditional patch attacks, and demonstrates unique effectiveness and practical value for attacking VLA feature spaces.

摘要: 近年来，物化智能中的视觉-语言-动作（VLA）模型发展迅速。然而，现有的对抗攻击方法需要昂贵的端到端训练，并且通常会生成明显的扰动补丁。为了解决这些限制，我们提出了ADVLA，这是一个直接对从视觉编码器投射到文本特征空间的特征应用对抗性扰动的框架。ADVLA在低幅度约束下有效扰乱下游动作预测，注意力引导允许扰动既集中又稀疏。我们引入了三种增强敏感性、加强稀疏性和集中扰动的策略。实验表明，在$L_{\infty}=4/255$约束下，ADVLA结合Top-K掩蔽修改的补丁数量不到10%，而攻击成功率接近100%。扰动集中在关键区域，在整个图像中几乎不可察觉，并且一步迭代只需大约0.06秒，显着优于传统的基于补丁的攻击。总而言之，ADVLA有效削弱了VLA模型在低幅度和局部稀疏条件下的下游动作预测，避免了传统补丁攻击的高训练成本和明显的扰动，在攻击VLA特征空间方面展现出独特的有效性和实用价值。



## **2. Multimodal Robust Prompt Distillation for 3D Point Cloud Models**

3D点云模型的多峰鲁棒即时蒸馏 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21574v1) [paper-pdf](https://arxiv.org/pdf/2511.21574v1)

**Authors**: Xiang Gu, Liming Lu, Xu Zheng, Anan Du, Yongbin Zhou, Shuchao Pang

**Abstract**: Adversarial attacks pose a significant threat to learning-based 3D point cloud models, critically undermining their reliability in security-sensitive applications. Existing defense methods often suffer from (1) high computational overhead and (2) poor generalization ability across diverse attack types. To bridge these gaps, we propose a novel yet efficient teacher-student framework, namely Multimodal Robust Prompt Distillation (MRPD) for distilling robust 3D point cloud model. It learns lightweight prompts by aligning student point cloud model's features with robust embeddings from three distinct teachers: a vision model processing depth projections, a high-performance 3D model, and a text encoder. To ensure a reliable knowledge transfer, this distillation is guided by a confidence-gated mechanism which dynamically balances the contribution of all input modalities. Notably, since the distillation is all during the training stage, there is no additional computational cost at inference. Extensive experiments demonstrate that MRPD substantially outperforms state-of-the-art defense methods against a wide range of white-box and black-box attacks, while even achieving better performance on clean data. Our work presents a new, practical paradigm for building robust 3D vision systems by efficiently harnessing multimodal knowledge.

摘要: 对抗性攻击对基于学习的3D点云模型构成重大威胁，严重削弱了它们在安全敏感应用程序中的可靠性。现有的防御方法通常存在以下问题：（1）计算负担高和（2）跨不同攻击类型的概括能力较差。为了弥合这些差距，我们提出了一种新颖而高效的师生框架，即多模式鲁棒即时蒸馏（MRPD），用于提取鲁棒的3D点云模型。它通过将学生点云模型的功能与来自三个不同教师的稳健嵌入相匹配来学习轻量级提示：处理深度投影的视觉模型、高性能3D模型和文本编码器。为了确保可靠的知识转移，这种提炼由信任门控机制指导，该机制动态平衡所有输入模式的贡献。值得注意的是，由于蒸馏都是在训练阶段进行的，因此推断时没有额外的计算成本。大量实验表明，MRPD在对抗各种白盒和黑匣子攻击方面的性能大大优于最先进的防御方法，甚至在干净的数据上实现了更好的性能。我们的工作提出了一种新的实用范式，用于通过有效利用多模式知识来构建稳健的3D视觉系统。



## **3. When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models**

当机器人遵守补丁时：通用可转移补丁对视觉-语言-动作模型的攻击 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21192v1) [paper-pdf](https://arxiv.org/pdf/2511.21192v1)

**Authors**: Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang

**Abstract**: Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.

摘要: 视觉-语言-动作（VLA）模型容易受到对抗性攻击，但普遍和可转移的攻击仍然未充分研究，因为大多数现有补丁过于适合单一模型，并且在黑匣子设置中失败。为了解决这一差距，我们对未知架构、微调变体和模拟到真实转变下针对VLA驱动机器人的通用、可转移的对抗补丁进行了系统研究。我们引入了UPA-RFAS（通过稳健特征、注意力和语义进行通用补丁攻击），这是一个统一框架，可以在共享特征空间中学习单个物理补丁，同时促进跨模型传输。UPA-RFAS结合了（i）特征空间目标与$\ell_1 $偏差先验和排斥性InfoNSO损失，以引发可转移的表示转移，（ii）鲁棒增强的两阶段最小-最大程序，其中内循环学习不可见的样本式扰动，外循环针对此硬化邻居优化通用补丁，以及（iii）两个特定于VLA的损失：补丁注意力支配性劫持文本$\到$视觉注意力，补丁语义失准以在没有标签的情况下引发图像与文本不匹配。跨不同VLA模型、操作套件和物理执行的实验表明，UPA-RFAS能够在模型、任务和观点之间一致地传输，暴露了实用的基于补丁的攻击表面，并为未来的防御建立了强大的基线。



## **4. CAHS-Attack: CLIP-Aware Heuristic Search Attack Method for Stable Diffusion**

CAHS攻击：用于稳定扩散的CLIP感知启发式搜索攻击方法 cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21180v1) [paper-pdf](https://arxiv.org/pdf/2511.21180v1)

**Authors**: Shuhan Xia, Jing Dai, Hui Ouyang, Yadong Shang, Dongxiao Zhao, Peipei Li

**Abstract**: Diffusion models exhibit notable fragility when faced with adversarial prompts, and strengthening attack capabilities is crucial for uncovering such vulnerabilities and building more robust generative systems. Existing works often rely on white-box access to model gradients or hand-crafted prompt engineering, which is infeasible in real-world deployments due to restricted access or poor attack effect. In this paper, we propose CAHS-Attack , a CLIP-Aware Heuristic Search attack method. CAHS-Attack integrates Monte Carlo Tree Search (MCTS) to perform fine-grained suffix optimization, leveraging a constrained genetic algorithm to preselect high-potential adversarial prompts as root nodes, and retaining the most semantically disruptive outcome at each simulation rollout for efficient local search. Extensive experiments demonstrate that our method achieves state-of-the-art attack performance across both short and long prompts of varying semantics. Furthermore, we find that the fragility of SD models can be attributed to the inherent vulnerability of their CLIP-based text encoders, suggesting a fundamental security risk in current text-to-image pipelines.

摘要: 当面对对抗提示时，扩散模型表现出显着的脆弱性，加强攻击能力对于发现此类脆弱性和构建更强大的生成系统至关重要。现有作品通常依赖于白盒访问模型梯度或手工制作的即时工程，由于访问受限或攻击效果不佳，这在现实世界的部署中是不可行的。在本文中，我们提出了CAHS-Attack，一种CLIP-Aware启发式搜索攻击方法。CAHS-Attack集成蒙特卡洛树搜索（MCTS）来执行细粒度的后缀优化，利用受约束的遗传算法预选高潜力的对抗性提示作为根节点，并在每次模拟推出时保留语义破坏性最强的结果，以进行高效的本地搜索。大量实验表明，我们的方法在不同语义的短提示和长提示中都实现了最先进的攻击性能。此外，我们发现SD模型的脆弱性可以归因于其基于CLIP的文本编码器的固有脆弱性，这表明当前文本到图像管道中存在根本的安全风险。



## **5. TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models**

TEAR：用于文本到视频模型的时间感知自动红色团队 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21145v1) [paper-pdf](https://arxiv.org/pdf/2511.21145v1)

**Authors**: Jiaming He, Guanyu Hou, Hongwei Li, Zhicong Huang, Kangjie Chen, Yi Yu, Wenbo Jiang, Guowen Xu, Tianwei Zhang

**Abstract**: Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%.

摘要: 文本转视频（T2 V）模型能够合成高质量、时间一致的动态视频内容，但多元化的一代本质上也带来了关键的安全挑战。现有的安全评估方法专注于静态图像和文本生成，不足以捕捉视频生成中复杂的时间动态。为了解决这个问题，我们提出了一个TEporal感知的自动化红色团队框架，名为TEAR，这是一个自动化框架，旨在发现专门与T2 V模型的动态时间排序相关的安全风险。TEAR采用一个通过两阶段方法优化的时间感知测试生成器：初始生成器训练和时间感知在线偏好学习，来制作文本无害的提示，利用时间动态来引发违反政策的视频输出。并采用细化模型循环提高即时隐形性和对抗有效性。广泛的实验评估证明了TEAR在开源和商用T2 V系统中的有效性，攻击成功率超过80%，比之前57%的最佳结果显着提高。



## **6. Securing the Model Context Protocol (MCP): Risks, Controls, and Governance**

保护模型上下文协议（HCP）：风险、控制和治理 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20920v1) [paper-pdf](https://arxiv.org/pdf/2511.20920v1)

**Authors**: Herman Errico, Jiquan Ngiam, Shanita Sojan

**Abstract**: The Model Context Protocol (MCP) replaces static, developer-controlled API integrations with more dynamic, user-driven agent systems, which also introduces new security risks. As MCP adoption grows across community servers and major platforms, organizations encounter threats that existing AI governance frameworks (such as NIST AI RMF and ISO/IEC 42001) do not yet cover in detail. We focus on three types of adversaries that take advantage of MCP s flexibility: content-injection attackers that embed malicious instructions into otherwise legitimate data; supply-chain attackers who distribute compromised servers; and agents who become unintentional adversaries by over-stepping their role. Based on early incidents and proof-of-concept attacks, we describe how MCP can increase the attack surface through data-driven exfiltration, tool poisoning, and cross-system privilege escalation. In response, we propose a set of practical controls, including per-user authentication with scoped authorization, provenance tracking across agent workflows, containerized sandboxing with input/output checks, inline policy enforcement with DLP and anomaly detection, and centralized governance using private registries or gateway layers. The aim is to help organizations ensure that unvetted code does not run outside a sandbox, tools are not used beyond their intended scope, data exfiltration attempts are detectable, and actions can be audited end-to-end. We close by outlining open research questions around verifiable registries, formal methods for these dynamic systems, and privacy-preserving agent operations.

摘要: 模型上下文协议（HCP）用更动态的用户驱动的代理系统取代了静态的、开发人员控制的API集成，这也带来了新的安全风险。随着社区服务器和主要平台上的社区服务器采用率的增长，组织遇到了现有人工智能治理框架（例如NIH AI RMF和ISO/IEC 42001）尚未详细涵盖的威胁。我们重点关注三种利用LCP灵活性的对手：将恶意指令嵌入到合法数据中的内容注入攻击者;分发受影响的服务器的供应链攻击者;以及通过越权而成为无意对手的代理。基于早期事件和概念验证攻击，我们描述了HCP如何通过数据驱动的溢出、工具中毒和跨系统特权升级来增加攻击面。作为回应，我们提出了一套实际的控制措施，包括具有范围授权的每用户身份验证、跨代理工作流程的出处跟踪、具有输入/输出检查的容器化沙箱、具有DLP和异常检测的内联策略执行，以及使用私有注册表或网关层的集中式治理。其目的是帮助组织确保已公开的代码不会在沙箱之外运行，工具的使用不会超出其预期范围，数据泄露尝试是可检测的，并且可以端到端审核操作。最后，我们概述了围绕可验证注册表、这些动态系统的正式方法以及隐私保护代理操作的开放研究问题。



## **7. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

对抗性混乱攻击：扰乱多模式大型语言模型 cs.CL

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20494v1) [paper-pdf](https://arxiv.org/pdf/2511.20494v1)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.

摘要: 我们引入了对抗性混乱攻击，这是针对多模式大型语言模型（MLLM）的一类新型威胁。与越狱或有针对性的错误分类不同，目标是引发系统性破坏，使模型生成不连贯或自信地错误的输出。应用程序包括将对抗图像嵌入到网站中，以防止MLLM支持的代理可靠运行。拟议的攻击使用一小部分开源MLLM来最大化下一个令牌的熵。在白盒设置中，我们表明，单个对抗图像可以扰乱集合中的所有模型，无论是在完整图像还是对抗验证码设置中。尽管依赖于基本的对抗技术（PVD），但攻击会产生转移到两个看不见的开源的扰动（例如，Qwen 3-DL）和专有（例如，GPT-5.1）型号。



## **8. Ranking-Enhanced Anomaly Detection Using Active Learning-Assisted Attention Adversarial Dual AutoEncoders**

使用主动学习辅助注意对抗性双AutoEnCoder进行排名增强异常检测 cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20480v1) [paper-pdf](https://arxiv.org/pdf/2511.20480v1)

**Authors**: Sidahmed Benabderrahmane, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a significant challenge in cybersecurity due to their stealthy and long-term nature. Modern supervised learning methods require extensive labeled data, which is often scarce in real-world cybersecurity environments. In this paper, we propose an innovative approach that leverages AutoEncoders for unsupervised anomaly detection, augmented by active learning to iteratively improve the detection of APT anomalies. By selectively querying an oracle for labels on uncertain or ambiguous samples, we minimize labeling costs while improving detection rates, enabling the model to improve its detection accuracy with minimal data while reducing the need for extensive manual labeling. We provide a detailed formulation of the proposed Attention Adversarial Dual AutoEncoder-based anomaly detection framework and show how the active learning loop iteratively enhances the model. The framework is evaluated on real-world imbalanced provenance trace databases produced by the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data. The datasets span multiple operating systems, including Android, Linux, BSD, and Windows, and cover two attack scenarios. The results have shown significant improvements in detection rates during active learning and better performance compared to other existing approaches.

摘要: 高级持续威胁（APT）因其隐蔽性和长期性而对网络安全构成重大挑战。现代监督学习方法需要大量的标记数据，而这些数据在现实世界的网络安全环境中通常是稀缺的。在本文中，我们提出了一种创新方法，该方法利用AutoEnCoders进行无监督异常检测，并通过主动学习来迭代改进APT异常检测。通过选择性地向Oracle查询不确定或模糊样本的标签，我们可以最大限度地降低标签成本，同时提高检测率，使模型能够用最少的数据提高检测准确性，同时减少对大量手动标签的需求。我们提供了拟议的基于注意力对抗双AutoEnCoder的异常检测框架的详细公式，并展示了主动学习循环如何迭代增强模型。该框架是在DARPA透明计算程序生成的现实世界不平衡来源追踪数据库上进行评估的，其中类似APT的攻击只占数据的0.004%。这些数据集跨越多个操作系统，包括Android、Linux、BDS和Windows，并涵盖两种攻击场景。结果表明，与其他现有方法相比，主动学习期间的检测率显着提高，并且性能更好。



## **9. Towards Trustworthy Wi-Fi Sensing: Systematic Evaluation of Deep Learning Model Robustness to Adversarial Attacks**

迈向可信的Wi-Fi感知：深度学习模型对抗性攻击鲁棒性的系统评估 cs.LG

19 pages, 8 figures, 7 tables

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20456v1) [paper-pdf](https://arxiv.org/pdf/2511.20456v1)

**Authors**: Shreevanth Krishnaa Gopalakrishnan, Stephen Hailes

**Abstract**: Machine learning has become integral to Channel State Information (CSI)-based human sensing systems and is expected to power applications such as device-free activity recognition and identity detection in future cellular and Wi-Fi generations. However, these systems rely on models whose decisions can be subtly perturbed, raising concerns for security and reliability in ubiquitous sensing. Quantifying and understanding the robustness of such models, defined as their ability to maintain accurate predictions under adversarial perturbations, is therefore critical before wireless sensing can be safely deployed in real-world environments.   This work presents a systematic evaluation of the robustness of CSI deep learning models under diverse threat models (white-box, black-box/transfer, and universal perturbations) and varying degrees of attack realism. We establish a framework to compare compact temporal autoencoder models with larger deep architectures across three public datasets, quantifying how model scale, training regime, and physical constraints influence robustness. Our experiments show that smaller models, while efficient and equally performant on clean data, are markedly less robust. We further confirm that physically realizable signal-space perturbations, designed to be feasible in real wireless channels, significantly reduce attack success compared to unconstrained feature-space attacks. Adversarial training mitigates these vulnerabilities, improving mean robust accuracy with only moderate degradation in clean performance across both model classes. As wireless sensing advances towards reliable, cross-domain operation, these findings provide quantitative baselines for robustness estimation and inform design principles for secure and trustworthy human-centered sensing systems.

摘要: 机器学习已成为基于通道状态信息（SI）的人类传感系统不可或缺的一部分，预计将为未来蜂窝和Wi-Fi一代的无设备活动识别和身份检测等应用提供动力。然而，这些系统依赖于其决策可能会受到微妙干扰的模型，这引发了人们对无处不在传感中的安全性和可靠性的担忧。因此，量化和理解此类模型的鲁棒性（定义为它们在对抗性扰动下保持准确预测的能力）在无线传感能够安全地部署在现实世界环境中之前至关重要。   这项工作对不同威胁模型（白盒、黑匣子/传输和普遍扰动）和不同程度的攻击现实性下的SI深度学习模型的稳健性进行了系统评估。我们建立了一个框架，将紧凑的时态自动编码器模型与三个公共数据集中的更大深度架构进行比较，量化模型规模、训练机制和物理约束如何影响稳健性。我们的实验表明，较小的模型虽然在干净的数据上高效且性能相同，但鲁棒性明显较差。我们进一步证实，与无约束特征空间攻击相比，物理上可实现的信号空间扰动（设计为在真实无线通道中可行）显着降低了攻击成功率。对抗性训练可以缓解这些漏洞，提高平均稳健准确性，但两个模型类别的干净性能仅适度下降。随着无线传感向可靠的跨域操作迈进，这些发现为稳健性估计提供了定量基线，并为安全且值得信赖的以人为本的传感系统的设计原则提供了信息。



## **10. V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs**

V-攻击：针对LVLM的可控对抗攻击的解纠缠价值特征 cs.CV

21 pages

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20223v1) [paper-pdf](https://arxiv.org/pdf/2511.20223v1)

**Authors**: Sen Nie, Jie Zhang, Jianxin Yan, Shiguang Shan, Xilin Chen

**Abstract**: Adversarial attacks have evolved from simply disrupting predictions on conventional task-specific models to the more complex goal of manipulating image semantics on Large Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image. We attribute this limitation to semantic entanglement in the patch-token representations on which adversarial attacks typically operate: global context aggregated by self-attention in the vision encoder dominates individual patch features, making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a key insight: value features (V) computed within the transformer attention block serve as much more precise handles for manipulation. We show that V suppresses global-context channels, allowing it to retain high-entropy, disentangled local semantic information. Building on this discovery, we propose V-Attack, a novel method designed for precise local semantic attacks. V-Attack targets the value features and introduces two core components: (1) a Self-Value Enhancement module to refine V's intrinsic semantic richness, and (2) a Text-Guided Value Manipulation module that leverages text prompts to locate source concept and optimize it toward a target concept. By bypassing the entangled patch features, V-Attack achieves highly effective semantic control. Extensive experiments across diverse LVLMs, including LLaVA, InternVL, DeepseekVL and GPT-4o, show that V-Attack improves the attack success rate by an average of 36% over state-of-the-art methods, exposing critical vulnerabilities in modern visual-language understanding. Our code and data are available https://github.com/Summu77/V-Attack.

摘要: 对抗性攻击已经从简单地破坏传统任务特定模型上的预测发展到在大型视觉语言模型（LVLM）上操纵图像语义的更复杂目标。然而，现有的方法难以控制，并且无法准确地操纵图像中特定概念的语义。我们将这种限制归因于对抗性攻击通常运作的补丁令牌表示中的语义纠缠：视觉编码器中由自我注意力聚集的全局上下文主导了单个补丁特征，使得它们对于精确的局部语义操纵来说不可靠。我们的系统性调查揭示了一个关键见解：在Transformer注意力块内计算的值特征（V）可以作为更精确的操纵处理。我们表明，V抑制了全球上下文通道，使其能够保留高熵、解开的局部语义信息。在这一发现的基础上，我们提出了V-Attack，这是一种专为精确的局部语义攻击而设计的新颖方法。V-Attack针对价值特征并引入了两个核心组件：（1）自我价值增强模块，用于细化V内在的语义丰富性，和（2）文本引导的价值操纵模块，利用文本提示来定位源概念并将其优化为目标概念。通过绕过纠缠补丁特征，V-Attack实现了高效的语义控制。在LLaVA、InternVLL、DeepseekVLL和GPT-4 o等各种LVLM上进行的广泛实验表明，V-Attack比最先进的方法平均提高了36%的攻击成功率，暴露了现代视觉语言理解中的关键漏洞。我们的代码和数据可访问https://github.com/Summu77/V-Attack。



## **11. On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation**

论通过一次扰动劫持MLLM决策链的可行性 cs.CV

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20002v1) [paper-pdf](https://arxiv.org/pdf/2511.20002v1)

**Authors**: Changyue Li, Jiaying Li, Youliang Yuan, Jiaming He, Zhicong Huang, Pinjia He

**Abstract**: Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks.   This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag".   To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame.

摘要: 传统的对抗攻击专注于操纵神经网络的单个决策。然而，现实世界的模型通常以一系列决策的方式运行，其中孤立的错误可以很容易地纠正，但连锁错误可能会导致严重的风险。   本文揭示了一种新颖的威胁：一个单一的扰动就可以劫持整个决策链。我们证明了将模型输出操作为多个预定义结果的可行性，例如同时将“非机动车道”标志误分类为“机动车道”，将“行人”误分类为“塑料袋”。   为了揭露这一威胁，我们引入了语义感知通用扰动（SAUP），它根据输入的语义引发不同的结果。我们通过开发一种有效的算法来克服优化挑战，该算法使用语义分离策略在规范化空间中搜索扰动。为了评估SAUP的实际威胁，我们提出了RIST，这是一个具有细粒度语义注释的新现实世界图像数据集。对三个多模式大型语言模型的广泛实验证明了它们的脆弱性，仅使用对抗框架控制五个不同目标时，攻击成功率达到70%。



## **12. Continual Audio Deepfake Detection via Universal Adversarial Perturbation**

通过普遍对抗扰动进行连续音频深度伪造检测 cs.SD

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.19974v1) [paper-pdf](https://arxiv.org/pdf/2511.19974v1)

**Authors**: Wangjie Li, Lin Li, Qingyang Hong

**Abstract**: The rapid advancement of speech synthesis and voice conversion technologies has raised significant security concerns in multimedia forensics. Although current detection models demonstrate impressive performance, they struggle to maintain effectiveness against constantly evolving deepfake attacks. Additionally, continually fine-tuning these models using historical training data incurs substantial computational and storage costs. To address these limitations, we propose a novel framework that incorporates Universal Adversarial Perturbation (UAP) into audio deepfake detection, enabling models to retain knowledge of historical spoofing distribution without direct access to past data. Our method integrates UAP seamlessly with pre-trained self-supervised audio models during fine-tuning. Extensive experiments validate the effectiveness of our approach, showcasing its potential as an efficient solution for continual learning in audio deepfake detection.

摘要: 语音合成和语音转换技术的快速发展引发了多媒体取证中的重大安全问题。尽管当前的检测模型表现出令人印象深刻的性能，但它们很难保持有效性来对抗不断发展的Deepfake攻击。此外，使用历史训练数据不断微调这些模型会产生巨大的计算和存储成本。为了解决这些局限性，我们提出了一种新颖的框架，将普遍对抗微扰（UAP）结合到音频深度伪造检测中，使模型能够保留历史欺骗分布的知识，而无需直接访问过去的数据。我们的方法在微调期间将UAP与预先训练的自我监督音频模型无缝集成。大量的实验验证了我们方法的有效性，展示了其作为音频深度伪造检测持续学习的有效解决方案的潜力。



## **13. Multi-Hypotheses Ego-Tracking for Resilient Navigation**

弹性导航的多假设自我跟踪 eess.SY

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19770v2) [paper-pdf](https://arxiv.org/pdf/2511.19770v2)

**Authors**: Peter Iwer Hoedt Karstensen, Roberto Galeazzi

**Abstract**: Autonomous robots relying on radio frequency (RF)-based localization such as global navigation satellite system (GNSS), ultra-wide band (UWB), and 5G integrated sensing and communication (ISAC) are vulnerable to spoofing and sensor manipulation. This paper presents a resilient navigation architecture that combines multi-hypothesis estimation with a Poisson binomial windowed-count detector for anomaly identification and isolation. A state machine coordinates transitions between operation, diagnosis, and mitigation, enabling adaptive response to adversarial conditions. When attacks are detected, trajectory re-planning based on differential flatness allows information-gathering maneuvers minimizing performance loss. Case studies demonstrate effective detection of biased sensors, maintenance of state estimation, and recovery of nominal operation under persistent spoofing attacks

摘要: 依赖于基于射频（RF）的定位的自主机器人，例如全球导航卫星系统（GNSS）、超宽带（UWB）和5G集成传感和通信（ISAC），很容易受到欺骗和传感器操纵。本文提出了一种弹性导航架构，该架构将多假设估计与Poisson二项窗口计数检测器相结合，用于异常识别和隔离。状态机协调操作、诊断和缓解之间的过渡，从而能够对对抗条件做出自适应响应。当检测到攻击时，基于差异平坦度的轨迹重新规划允许信息收集操作最大限度地减少性能损失。案例研究展示了在持续欺骗攻击下有效检测有偏差的传感器、维持状态估计以及恢复名义操作



## **14. Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?**

受神经启发的多模式视觉语言模型能否抵御会员推断隐私泄露？ cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.20710v1) [paper-pdf](https://arxiv.org/pdf/2511.20710v1)

**Authors**: David Amebley, Sayanton Dibbo

**Abstract**: In the age of agentic AI, the growing deployment of multi-modal models (MMs) has introduced new attack vectors that can leak sensitive training data in MMs, causing privacy leakage. This paper investigates a black-box privacy attack, i.e., membership inference attack (MIA) on multi-modal vision-language models (VLMs). State-of-the-art research analyzes privacy attacks primarily to unimodal AI-ML systems, while recent studies indicate MMs can also be vulnerable to privacy attacks. While researchers have demonstrated that biologically inspired neural network representations can improve unimodal model resilience against adversarial attacks, it remains unexplored whether neuro-inspired MMs are resilient against privacy attacks. In this work, we introduce a systematic neuroscience-inspired topological regularization (tau) framework to analyze MM VLMs resilience against image-text-based inference privacy attacks. We examine this phenomenon using three VLMs: BLIP, PaliGemma 2, and ViT-GPT2, across three benchmark datasets: COCO, CC3M, and NoCaps. Our experiments compare the resilience of baseline and neuro VLMs (with topological regularization), where the tau > 0 configuration defines the NEURO variant of VLM. Our results on the BLIP model using the COCO dataset illustrate that MIA attack success in NEURO VLMs drops by 24% mean ROC-AUC, while achieving similar model utility (similarities between generated and reference captions) in terms of MPNet and ROUGE-2 metrics. This shows neuro VLMs are comparatively more resilient against privacy attacks, while not significantly compromising model utility. Our extensive evaluation with PaliGemma 2 and ViT-GPT2 models, on two additional datasets: CC3M and NoCaps, further validates the consistency of the findings. This work contributes to the growing understanding of privacy risks in MMs and provides evidence on neuro VLMs privacy threat resilience.

摘要: 在代理人工智能时代，多模式模型（NPS）的不断增加的部署引入了新的攻击载体，这些攻击载体可以泄露NPS中的敏感训练数据，从而导致隐私泄露。本文研究了黑匣子隐私攻击，即对多模式视觉语言模型（VLMS）的成员资格推理攻击（MIA）。最先进的研究主要分析了针对单模式AI-ML系统的隐私攻击，而最近的研究表明，收件箱也容易受到隐私攻击。虽然研究人员已经证明，生物启发的神经网络表示可以提高单峰模型抵御对抗性攻击的弹性，但神经启发的NPS是否能够抵御隐私攻击仍然有待探讨。在这项工作中，我们引入了一个系统性的神经科学启发的拓学正规化（tau）框架来分析MM VLM针对基于图像-文本的推理隐私攻击的弹性。我们在三个基准数据集（COCO、CC 3 M和NoCaps）上使用三个VLM（BLIP、PaliGemma 2和ViT-GPT 2）来检查这种现象。我们的实验比较了基线和神经VLM（具有拓学规则化）的弹性，其中tau > 0配置定义了VLM的NEURO变体。我们使用COCO数据集对BLIP模型的结果表明，NEURO VLM中的MIA攻击成功率平均下降了24%，同时在MPNet和ROUGE-2指标方面实现了类似的模型效用（生成的和参考字幕之间的相似性）。这表明神经元VLM对隐私攻击的弹性相对更强，同时不会显着损害模型的实用性。我们使用PaliGemma 2和ViT-GPT 2模型对另外两个数据集（CC 3 M和NoCaps）进行了广泛评估，进一步验证了结果的一致性。这项工作有助于加深人们对收件箱中隐私风险的了解，并提供了有关神经元VLM隐私威胁复原力的证据。



## **15. Synthetic Data: AI's New Weapon Against Android Malware**

合成数据：人工智能对抗Android恶意软件的新武器 cs.CR

23 pages, 18 figures, 8 tables. Accepted for publication at the JBCS

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19649v1) [paper-pdf](https://arxiv.org/pdf/2511.19649v1)

**Authors**: Angelo Gaspar Diniz Nogueira, Kayua Oleques Paim, Hendrio Bragança, Rodrigo Brandão Mansilha, Diego Kreutz

**Abstract**: The ever-increasing number of Android devices and the accelerated evolution of malware, reaching over 35 million samples by 2024, highlight the critical importance of effective detection methods. Attackers are now using Artificial Intelligence to create sophisticated malware variations that can easily evade traditional detection techniques. Although machine learning has shown promise in malware classification, its success relies heavily on the availability of up-to-date, high-quality datasets. The scarcity and high cost of obtaining and labeling real malware samples presents significant challenges in developing robust detection models. In this paper, we propose MalSynGen, a Malware Synthetic Data Generation methodology that uses a conditional Generative Adversarial Network (cGAN) to generate synthetic tabular data. This data preserves the statistical properties of real-world data and improves the performance of Android malware classifiers. We evaluated the effectiveness of this approach using various datasets and metrics that assess the fidelity of the generated data, its utility in classification, and the computational efficiency of the process. Our experiments demonstrate that MalSynGen can generalize across different datasets, providing a viable solution to address the issues of obsolescence and low quality data in malware detection.

摘要: Android设备数量的不断增加和恶意软件的加速演变，到2024年样本数量将超过3500万个，凸显了有效检测方法的至关重要性。攻击者现在正在使用人工智能创建复杂的恶意软件变体，可以轻松逃避传统检测技术。尽管机器学习在恶意软件分类方面表现出了希望，但其成功在很大程度上依赖于最新、高质量数据集的可用性。获取和标记真实恶意软件样本的稀缺性和高成本给开发稳健的检测模型带来了巨大挑战。在本文中，我们提出了MalSynGen，这是一种恶意软件合成数据生成方法，它使用条件生成对抗网络（cGAN）来生成合成表格数据。这些数据保留了真实数据的统计属性，并提高了Android恶意软件分类器的性能。我们使用各种数据集和指标评估了这种方法的有效性，这些数据集和指标评估了生成数据的保真度、其在分类中的实用性以及该过程的计算效率。我们的实验表明，MalSynGen可以在不同的数据集上进行推广，为解决恶意软件检测中的过时和低质量数据问题提供了一个可行的解决方案。



## **16. Targeted Manipulation: Slope-Based Attacks on Financial Time-Series Data**

有针对性的操纵：对金融时间序列数据的基于斜率的攻击 cs.LG

13 pages, 6 figures, 4 tables, preprint; Total including Appendix: 21 pages, 11 figures, 7 tables

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19330v1) [paper-pdf](https://arxiv.org/pdf/2511.19330v1)

**Authors**: Dominik Luszczynski

**Abstract**: A common method of attacking deep learning models is through adversarial attacks, which occur when an attacker specifically modifies the input of a model to produce an incorrect result. Adversarial attacks have been deeply investigated in the image domain; however, there is less research in the time-series domain and very little for forecasting financial data. To address these concerns, this study aims to build upon previous research on adversarial attacks for time-series data by introducing two new slope-based methods aimed to alter the trends of the predicted stock forecast generated by an N-HiTS model. Compared to the normal N-HiTS predictions, the two new slope-based methods, the General Slope Attack and Least-Squares Slope Attack, can manipulate N-HiTS predictions by doubling the slope. These new slope attacks can bypass standard security mechanisms, such as a discriminator that filters real and perturbed inputs, reducing a 4-layered CNN's specificity to 28% and accuracy to 57%. Furthermore, the slope based methods were incorporated into a GAN architecture as a means of generating realistic synthetic data, while simultaneously fooling the model. Finally, this paper also proposes a sample malware designed to inject an adversarial attack in the model inference library, proving that ML-security research should not only focus on making the model safe, but also securing the entire pipeline.

摘要: 攻击深度学习模型的一种常见方法是通过对抗攻击，当攻击者专门修改模型的输入以产生错误的结果时，就会发生对抗攻击。对抗性攻击在图像领域得到了深入的研究;然而，时间序列领域的研究较少，预测金融数据的研究也很少。为了解决这些问题，这项研究旨在以之前关于时间序列数据对抗攻击的研究为基础，引入两种新的基于斜坡的方法，旨在改变N-HiTS模型生成的预测股票预测的趋势。与正常的N-HiTS预测相比，两种新的基于斜坡的方法（一般斜坡攻击和最小平方斜坡攻击）可以通过将斜坡加倍来操纵N-HiTS预测。这些新的斜坡攻击可以绕过标准的安全机制，例如过滤真实和受干扰输入的收件箱，将4层CNN的特异性降低到28%，准确性降低到57%。此外，基于斜坡的方法被整合到GAN架构中，作为生成真实合成数据的一种手段，同时愚弄模型。最后，本文还提出了一个旨在在模型推理库中注入对抗性攻击的恶意软件样本，证明ML安全研究不仅应该关注于使模型安全，还应该关注于确保整个管道的安全。



## **17. Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation**

美杜莎：对多模式医学检索增强一代的跨模式可转移对抗攻击 cs.CR

Accepted at KDD 2026 First Cycle (full version). Authors marked with * contributed equally. Yi Liu is the lead author

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19257v1) [paper-pdf](https://arxiv.org/pdf/2511.19257v1)

**Authors**: Yingjia Shang, Yi Liu, Huimin Wang, Furong Li, Wenfang Sun, Wu Chengyu, Yefeng Zheng

**Abstract**: With the rapid advancement of retrieval-augmented vision-language models, multimodal medical retrieval-augmented generation (MMed-RAG) systems are increasingly adopted in clinical decision support. These systems enhance medical applications by performing cross-modal retrieval to integrate relevant visual and textual evidence for tasks, e.g., report generation and disease diagnosis. However, their complex architecture also introduces underexplored adversarial vulnerabilities, particularly via visual input perturbations. In this paper, we propose Medusa, a novel framework for crafting cross-modal transferable adversarial attacks on MMed-RAG systems under a black-box setting. Specifically, Medusa formulates the attack as a perturbation optimization problem, leveraging a multi-positive InfoNCE loss (MPIL) to align adversarial visual embeddings with medically plausible but malicious textual targets, thereby hijacking the retrieval process. To enhance transferability, we adopt a surrogate model ensemble and design a dual-loop optimization strategy augmented with invariant risk minimization (IRM). Extensive experiments on two real-world medical tasks, including medical report generation and disease diagnosis, demonstrate that Medusa achieves over 90% average attack success rate across various generation models and retrievers under appropriate parameter configuration, while remaining robust against four mainstream defenses, outperforming state-of-the-art baselines. Our results reveal critical vulnerabilities in the MMed-RAG systems and highlight the necessity of robustness benchmarking in safety-critical medical applications. The code and data are available at https://anonymous.4open.science/r/MMed-RAG-Attack-F05A.

摘要: 随着检索增强视觉语言模型的快速发展，多模式医疗检索增强生成（MMed-RAG）系统越来越多地被用于临床决策支持。这些系统通过执行跨模式检索来集成任务的相关视觉和文本证据来增强医疗应用，例如，报告生成和疾病诊断。然而，它们复杂的架构也引入了未充分探索的对抗漏洞，特别是通过视觉输入扰动。在本文中，我们提出了MedUSA，这是一种新颖的框架，用于在黑匣子环境下对MMed-RAG系统进行跨模式可转移对抗攻击。具体来说，MedUSA将攻击描述为扰动优化问题，利用多正InfoNSO损失（MPIL）将对抗性视觉嵌入与医学上看似合理但恶意的文本目标对齐，从而劫持检索过程。为了增强可移植性，我们采用了代理模型集成，并设计了一种以不变风险最小化（RST）为基础的双环优化策略。对两项现实世界医疗任务（包括医疗报告生成和疾病诊断）的广泛实验表明，在适当的参数配置下，美杜莎在各种代模型和检索器中实现了超过90%的平均攻击成功率，同时对四种主流防御保持稳健，表现优于最先进的基线。我们的结果揭示了MMed-RAG系统中的关键漏洞，并强调了安全关键医疗应用中稳健性基准测试的必要性。代码和数据可在https://anonymous.4open.science/r/MMed-RAG-Attack-F05A上获取。



## **18. Adversarial Patch Attacks on Vision-Based Cargo Occupancy Estimation via Differentiable 3D Simulation**

通过差异化3D模拟对基于视觉的货物占有率估计的对抗补丁攻击 cs.CV

9 pages, 5 figures, 1 algorithm

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19254v1) [paper-pdf](https://arxiv.org/pdf/2511.19254v1)

**Authors**: Mohamed Rissal Hedna, Sesugh Samuel Nder

**Abstract**: Computer vision systems are increasingly adopted in modern logistics operations, including the estimation of trailer occupancy for planning, routing, and billing. Although effective, such systems may be vulnerable to physical adversarial attacks, particularly adversarial patches that can be printed and placed on interior surfaces. In this work, we study the feasibility of such attacks on a convolutional cargo-occupancy classifier using fully simulated 3D environments. Using Mitsuba 3 for differentiable rendering, we optimize patch textures across variations in geometry, lighting, and viewpoint, and compare their effectiveness to a 2D compositing baseline. Our experiments demonstrate that 3D-optimized patches achieve high attack success rates, especially in a denial-of-service scenario (empty to full), where success reaches 84.94 percent. Concealment attacks (full to empty) prove more challenging but still reach 30.32 percent. We analyze the factors influencing attack success, discuss implications for the security of automated logistics pipelines, and highlight directions for strengthening physical robustness. To our knowledge, this is the first study to investigate adversarial patch attacks for cargo-occupancy estimation in physically realistic, fully simulated 3D scenes.

摘要: 现代物流运营中越来越多地采用计算机视觉系统，包括估计拖车占用率以进行规划、路线和计费。尽管有效，但此类系统可能容易受到物理对抗攻击，特别是可以打印并放置在内表面上的对抗补丁。在这项工作中，我们研究了使用完全模拟的3D环境对卷积货物占有分类器进行此类攻击的可行性。使用Mitsuba 3进行可区分渲染，我们在几何、照明和视角的变化中优化补丁纹理，并将其有效性与2D合成基线进行比较。我们的实验表明，3D优化补丁可以实现很高的攻击成功率，特别是在拒绝服务场景（空到满）中，成功率达到84.94%。事实证明，隐藏攻击（从满到空）更具挑战性，但仍达到30.32%。我们分析了影响攻击成功的因素，讨论了对自动化物流管道安全性的影响，并强调了加强物理稳健性的方向。据我们所知，这是第一项在物理真实、完全模拟的3D场景中调查对抗补丁攻击以估计货物占用率的研究。



## **19. FedPoisonTTP: A Threat Model and Poisoning Attack for Federated Test-Time Personalization**

FedPoisonTTP：用于联邦测试时个性化的威胁模型和中毒攻击 cs.CR

13 pages, 3 figures, 2 tables

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19248v1) [paper-pdf](https://arxiv.org/pdf/2511.19248v1)

**Authors**: Md Akil Raihan Iftee, Syed Md. Ahnaf Hasan, Amin Ahsan Ali, AKM Mahbubur Rahman, Sajib Mistry, Aneesh Krishna

**Abstract**: Test-time personalization in federated learning enables models at clients to adjust online to local domain shifts, enhancing robustness and personalization in deployment. Yet, existing federated learning work largely overlooks the security risks that arise when local adaptation occurs at test time. Heterogeneous domain arrivals, diverse adaptation algorithms, and limited cross-client visibility create vulnerabilities where compromised participants can craft poisoned inputs and submit adversarial updates that undermine both global and per-client performance. To address this threat, we introduce FedPoisonTTP, a realistic grey-box attack framework that explores test-time data poisoning in the federated adaptation setting. FedPoisonTTP distills a surrogate model from adversarial queries, synthesizes in-distribution poisons using feature-consistency, and optimizes attack objectives to generate high-entropy or class-confident poisons that evade common adaptation filters. These poisons are injected during local adaptation and spread through collaborative updates, leading to broad degradation. Extensive experiments on corrupted vision benchmarks show that compromised participants can substantially diminish overall test-time performance.

摘要: 联合学习中的测试时个性化使客户端的模型能够在线调整以适应本地域变化，增强部署的稳健性和个性化。然而，现有的联邦学习工作在很大程度上忽视了测试时进行本地适应时出现的安全风险。不同的域到达、不同的适应算法和有限的跨客户端可见性会产生漏洞，受影响的参与者可以制作有毒的输入并提交对抗性更新，从而损害全球和每个客户端的性能。为了解决这一威胁，我们引入了FedPoisonTTP，这是一个现实的灰箱攻击框架，它探索联邦适应环境中的测试时数据中毒。FedPoisonTTP从对抗性查询中提取代理模型，使用特征一致性合成分布内毒药，并优化攻击目标以生成逃避常见适应过滤器的高熵或类别自信毒药。这些毒素在当地适应期间注入，并通过协作更新传播，导致广泛降解。针对受损视觉基准的大量实验表明，受损的参与者可能会大幅降低整体测试时表现。



## **20. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

通过树群双感知搜索和优化实现LLM安全性调整的对抗性攻击-防御协同进化 cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19218v2) [paper-pdf](https://arxiv.org/pdf/2511.19218v2)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.

摘要: 大型语言模型（LLM）在网络服务中迅速发展，提供了前所未有的能力，同时放大了社会风险。现有的作品往往专注于孤立的越狱攻击或静态防御，忽视了现实世界网络环境中不断变化的威胁与保障措施之间的动态相互作用。为了缓解这些挑战，我们提出了ACE安全（针对LLM安全的对抗协同进化），一个新颖的框架，通过无缝集成两个关键的创新过程来联合优化攻击和防御模型：（1）群体感知策略引导的蒙特卡洛树搜索（GS-MCTS），它有效地探索越狱策略以发现漏洞并生成多样化的对抗样本;（2）对抗性课程树感知群组策略优化（AC-TSYS），通过课程强化学习，利用具有挑战性的样本联合训练攻击和防御LLM，实现稳健的相互改进。多个基准的评估表明，我们的方法优于现有的攻击和防御方法，并为开发能够可持续支持负责任的人工智能生态系统的LLM提供了可行的途径。



## **21. Learning to Compress Graphs via Dual Agents for Consistent Topological Robustness Evaluation**

学习通过双代理压缩图以进行一致的布局鲁棒性评估 cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.18958v2) [paper-pdf](https://arxiv.org/pdf/2511.18958v2)

**Authors**: Qisen Chai, Yansong Wang, Junjie Huang, Tao Jia

**Abstract**: As graph-structured data grow increasingly large, evaluating their robustness under adversarial attacks becomes computationally expensive and difficult to scale. To address this challenge, we propose to compress graphs into compact representations that preserve both topological structure and robustness profile, enabling efficient and reliable evaluation. We propose Cutter, a dual-agent reinforcement learning framework composed of a Vital Detection Agent (VDA) and a Redundancy Detection Agent (RDA), which collaboratively identify structurally vital and redundant nodes for guided compression. Cutter incorporates three key strategies to enhance learning efficiency and compression quality: trajectory-level reward shaping to transform sparse trajectory returns into dense, policy-equivalent learning signals; prototype-based shaping to guide decisions using behavioral patterns from both high- and low-return trajectories; and cross-agent imitation to enable safer and more transferable exploration. Experiments on multiple real-world graphs demonstrate that Cutter generates compressed graphs that retain essential static topological properties and exhibit robustness degradation trends highly consistent with the original graphs under various attack scenarios, thereby significantly improving evaluation efficiency without compromising assessment fidelity.

摘要: 随着图形结构数据变得越来越大，评估其在对抗攻击下的稳健性变得计算昂贵且难以扩展。为了应对这一挑战，我们建议将图压缩为紧凑的表示，以保留拓扑结构和鲁棒性轮廓，从而实现高效可靠的评估。我们提出了Cutter，这是一个由重要检测代理（VDA）和冗余检测代理（RDA）组成的双代理强化学习框架，它可以协作识别结构上重要的和冗余的节点以进行引导压缩。Cutter结合了三项关键策略来提高学习效率和压缩质量：专家级奖励整形，将稀疏轨迹回报转化为密集的、政策等效的学习信号;基于原型的整形，使用高回报和低回报轨迹的行为模式来指导决策;和跨代理模仿，以实现更安全、更可转移的探索。对多个现实世界图的实验表明，Cutter生成的压缩图保留了基本的静态拓扑属性，并在各种攻击场景下表现出与原始图高度一致的鲁棒性退化趋势，从而显着提高了评估效率，而不会损害评估保真度。



## **22. Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations**

以负责任的人工智能考虑保护大型语言模型免受越狱利用 cs.CR

20 pages including appendix; technical report; NeurIPS 2024 style

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18933v1) [paper-pdf](https://arxiv.org/pdf/2511.18933v1)

**Authors**: Ryan Wong, Hosea David Yu Fei Ng, Dhananjai Sharma, Glenn Jun Jie Ng, Kavishvaran Srinivasan

**Abstract**: Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project

摘要: 大型语言模型（LLM）仍然容易受到越狱漏洞利用的影响，这些漏洞绕过安全过滤器并引发有害或不道德行为。这项工作对预算级、模型级和训练时干预措施的现有越狱防御进行了系统分类，然后提出了三种拟议的防御策略。首先，预算级防御框架通过净化、重述和自适应系统防护来检测并中和对抗输入。其次，基于日志的转向防御通过安全敏感层中的推理时间载体转向来加强拒绝行为。第三，领域特定代理防御采用MetaGPT框架来实施结构化的、基于角色的协作和领域遵守。对基准数据集的实验显示，攻击成功率大幅降低，在基于代理的防御下实现了全面缓解。总体而言，这项研究强调了越狱如何对LLM构成重大安全威胁，并确定了预防的关键干预点，同时指出防御策略通常涉及安全性、性能和可扩展性之间的权衡。代码可访问：https://github.com/Kuro0911/CS5446-Project



## **23. BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models**

BackdoorVLM：视觉语言模型后门攻击的基准 cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18921v1) [paper-pdf](https://arxiv.org/pdf/2511.18921v1)

**Authors**: Juncheng Li, Yige Li, Hanxun Huang, Yunhao Chen, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .

摘要: 后门攻击通过注入可能在推理时被恶意激活的隐藏行为来破坏机器学习系统的可靠性和可信性。虽然此类威胁在单模式环境中得到了广泛研究，但它们对多模式基础模型（尤其是视觉语言模型（VLM）的影响在很大程度上仍然没有得到充分的研究。在这项工作中，我们引入了\textBF{BackdoorVLM}，这是第一个用于在广泛的设置中系统评估对VLM的后门攻击的全面基准。它采用统一的视角，在核心视觉语言任务（包括图像字幕和视觉问答）中注入和分析后门。BackdoorVLM将多模式后门威胁分为5个代表性类别：定向拒绝、恶意注入、越狱、概念替代和感知劫持。每个类别都捕获了对手可以操纵模型行为的独特途径。我们使用涵盖文本、图像和双峰触发器的12种代表性攻击方法来评估这些威胁，并在2个开源VLM和3个多模式数据集上进行了测试。我们的分析表明，VLM对文本指令表现出很强的敏感性，并且在双峰后门中，文本触发器在形成后门映射时通常会触发图像触发器。值得注意的是，涉及文本形式的后门仍然非常有效，中毒率低至1%，大多数任务的成功率超过90%。这些发现凸显了当前VLM中先前未充分探索的重大漏洞。我们希望BackdoorVLM能够成为分析和缓解多模式后门威胁的有用基准。代码可访问：https://github.com/bin015/BackdoorVLM。



## **24. EAGER: Edge-Aligned LLM Defense for Robust, Efficient, and Accurate Cybersecurity Question Answering**

EAGER：边缘对齐的LLM防御，实现强大、高效和准确的网络安全问题解答 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19523v1) [paper-pdf](https://arxiv.org/pdf/2511.19523v1)

**Authors**: Onat Gungor, Roshan Sood, Jiasheng Zhou, Tajana Rosing

**Abstract**: Large Language Models (LLMs) are highly effective for cybersecurity question answering (QA) but are difficult to deploy on edge devices due to their size. Quantization reduces memory and compute requirements but often degrades accuracy and increases vulnerability to adversarial attacks. We present EAGER, an edge-aligned defense framework that integrates parameter-efficient quantization with domain-specific preference alignment to jointly optimize efficiency, robustness, and accuracy. Unlike prior methods that address these aspects separately, EAGER leverages Quantized Low-Rank Adaptation (QLoRA) for low-cost fine-tuning and Direct Preference Optimization (DPO) on a self-constructed cybersecurity preference dataset, eliminating the need for human labels. Experiments show that EAGER reduces adversarial attack success rates by up to 7.3x and improves QA accuracy by up to 55% over state-of-the-art defenses, while achieving the lowest response latency on a Jetson Orin, demonstrating its practical edge deployment.

摘要: 大型语言模型（LLM）对于网络安全问题回答（QA）非常有效，但由于其尺寸而难以在边缘设备上部署。量化降低了内存和计算需求，但通常会降低准确性并增加对对抗攻击的脆弱性。我们提出了EAGER，这是一个边缘对齐的防御框架，它将参数高效量化与特定领域的偏好对齐集成在一起，以共同优化效率、稳健性和准确性。与单独解决这些方面的现有方法不同，EAGER利用量化低等级适应（QLoRA）对自构建的网络安全偏好数据集进行低成本微调和直接偏好优化（DPO），消除了对人类标签的需求。实验表明，与最先进的防御相比，EAGER将对抗攻击成功率降低了7.3倍，将QA准确性提高了55%，同时在Jetson Orin上实现了最低的响应延迟，展示了其实用的边缘部署。



## **25. Now You See It, Now You Don't - Instant Concept Erasure for Safe Text-to-Image and Video Generation**

现在你看到了，现在你不看到-即时概念擦除，实现安全的文本到图像和视频生成 cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18684v1) [paper-pdf](https://arxiv.org/pdf/2511.18684v1)

**Authors**: Shristi Das Biswas, Arani Roy, Kaushik Roy

**Abstract**: Robust concept removal for text-to-image (T2I) and text-to-video (T2V) models is essential for their safe deployment. Existing methods, however, suffer from costly retraining, inference overhead, or vulnerability to adversarial attacks. Crucially, they rarely model the latent semantic overlap between the target erase concept and surrounding content -- causing collateral damage post-erasure -- and even fewer methods work reliably across both T2I and T2V domains. We introduce Instant Concept Erasure (ICE), a training-free, modality-agnostic, one-shot weight modification approach that achieves precise, persistent unlearning with zero overhead. ICE defines erase and preserve subspaces using anisotropic energy-weighted scaling, then explicitly regularises against their intersection using a unique, closed-form overlap projector. We pose a convex and Lipschitz-bounded Spectral Unlearning Objective, balancing erasure fidelity and intersection preservation, that admits a stable and unique analytical solution. This solution defines a dissociation operator that is translated to the model's text-conditioning layers, making the edit permanent and runtime-free. Across targeted removals of artistic styles, objects, identities, and explicit content, ICE efficiently achieves strong erasure with improved robustness to red-teaming, all while causing only minimal degradation of original generative abilities in both T2I and T2V models.

摘要: 文本到图像（T2 I）和文本到视频（T2 V）模型的稳健概念删除对于它们的安全部署至关重要。然而，现有的方法存在成本高昂的再培训、推理费用或容易受到对抗攻击的影响。至关重要的是，他们很少对目标擦除概念和周围内容之间潜在的语义重叠进行建模，从而在擦除后造成附带损害，而且在T2 I和T2 V域中可靠工作的方法甚至更少。我们引入即时概念擦除（ICE），这是一种免训练、模式不可知、一次性权重修改方法，可以以零费用实现精确、持续的去学习。ICE使用各向异性能量加权缩放来定义擦除和保留子空间，然后使用独特的封闭形式重叠投影仪针对它们的相交进行显式调整。我们提出了一个凸的和利普希茨有界的光谱取消学习目标，平衡了擦除保真度和交集保留，它允许稳定且独特的分析解。该解决方案定义了一个分离操作符，该操作符被转换为模型的文本条件层，使编辑永久且不受运行时限制。在有针对性地删除艺术风格、对象、身份和显式内容时，ICE有效地实现了强擦除，并提高了对红色团队的鲁棒性，同时在T2 I和T2 V模型中仅导致原始生成能力的最小退化。



## **26. Robust Physical Adversarial Patches Using Dynamically Optimized Clusters**

使用动态优化集群的稳健物理对抗补丁 cs.CV

Supplementary material available at: https://drive.google.com/drive/folders/1Yntcc9CARdbvoJJ51cyUm1DWGSvU9X4V?usp=drive_link

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18656v1) [paper-pdf](https://arxiv.org/pdf/2511.18656v1)

**Authors**: Harrison Bagley, Will Meakin, Simon Lucey, Yee Wei Law, Tat-Jun Chin

**Abstract**: Physical adversarial attacks on deep learning systems is concerning due to the ease of deploying such attacks, usually by placing an adversarial patch in a scene to manipulate the outcomes of a deep learning model. Training such patches typically requires regularization that improves physical realizability (e.g., printability, smoothness) and/or robustness to real-world variability (e.g. deformations, viewing angle, noise). One type of variability that has received little attention is scale variability. When a patch is rescaled, either digitally through downsampling/upsampling or physically through changing imaging distances, interpolation-induced color mixing occurs. This smooths out pixel values, resulting in a loss of high-frequency patterns and degrading the adversarial signal. To address this, we present a novel superpixel-based regularization method that guides patch optimization to scale-resilient structures. Our ap proach employs the Simple Linear Iterative Clustering (SLIC) algorithm to dynamically cluster pixels in an adversarial patch during optimization. The Implicit Function Theorem is used to backpropagate gradients through SLIC to update the superpixel boundaries and color. This produces patches that maintain their structure over scale and are less susceptible to interpolation losses. Our method achieves greater performance in the digital domain, and when realized physically, these performance gains are preserved, leading to improved physical performance. Real-world performance was objectively assessed using a novel physical evaluation protocol that utilizes screens and cardboard cut-outs to systematically vary real-world conditions.

摘要: 对深度学习系统的物理对抗攻击令人担忧，因为此类攻击很容易部署，通常是通过在场景中放置对抗补丁来操纵深度学习模型的结果。训练此类补丁通常需要提高物理实现性的正规化（例如，可印刷性、平滑度）和/或对现实世界可变性（例如变形、视角、噪音）的鲁棒性。一种很少受到关注的变异性是规模变异性。当补丁重新缩放时，无论是通过下采样/上采样进行数字调整，还是通过改变成像距离进行物理调整，都会发生内插引起的颜色混合。这会平滑像素值，导致高频模式的丢失并降低对抗信号的性能。为了解决这个问题，我们提出了一种新型的基于超像素的正规化方法，该方法将补丁优化为具有规模弹性的结构。我们的方法采用简单线性迭代集群（SIIC）算法在优化期间动态集群对抗补丁中的像素。隐函数定理用于通过SIIC反向传播梯度，以更新超像素边界和颜色。这产生的补丁可以超规模地维持其结构，并且不太容易受到插值损失的影响。我们的方法在数字领域实现了更高的性能，并且当物理实现时，这些性能收益会被保留，从而改善物理性能。使用一种新型的物理评估方案客观评估现实世界的表现，该方案利用屏幕和纸板切口来系统性地改变现实世界的条件。



## **27. Algorithmic detection of false data injection attacks in cyber-physical systems**

网络物理系统中虚假数据注入攻击的计算机检测 math.OC

13 pages, 6 figures

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18588v1) [paper-pdf](https://arxiv.org/pdf/2511.18588v1)

**Authors**: Souvik Das, Avishek Ghosh, Debasish Chatterjee

**Abstract**: This article introduces an anomaly detection based algorithm (AD-CPS) to detect false data injection attacks that fall under the category of data deception/integrity attacks, but with arbitrary information structure, in cyber-physical systems (CPSs) modeled as stochastic linear time-invariant systems. The core idea of this data-driven algorithm is based on the fact that an honest state (one not compromised by adversaries) generated by the CPS should concentrate near its weighted empirical mean of the immediate past samples. As the first theoretical result, we provide non-asymptotic guarantees on the false positive error incurred by the algorithm for attacks that are 2-step honest, referring to adversaries that act intermittently rather than successively. Moreover, we establish that for adversaries possessing a certain minimum energy, the false negative error incurred by AD-CPS is low. Extensive experiments were conducted on partially observed stochastic LTI systems to demonstrate these properties and to quantitatively compare AD-CPS with an optimal CUSUM-based test.

摘要: 本文介绍了一种基于异常检测的算法（AD-CPS）来检测虚假数据注入攻击，属于数据欺骗/完整性攻击的范畴，但具有任意的信息结构，在网络物理系统（CPS）建模为随机线性时不变系统。这种数据驱动算法的核心思想是基于这样一个事实，即CPS生成的诚实状态（不受对手损害的状态）应该集中在其最近过去样本的加权经验平均值附近。作为第一个理论结果，我们提供了非渐近保证的假阳性错误算法的攻击是2步诚实，指的是对手的行为间歇性，而不是连续的。此外，我们建立了具有一定的最小能量的对手，由AD-CPS引起的假阴性错误是低的。在部分观察到的随机LTI系统上进行了大量实验，以证明这些特性，并将AD-CPS与最佳基于CLARUM的测试进行定量比较。



## **28. Future-Back Threat Modeling: A Foresight-Driven Security Framework**

未来威胁建模：前瞻性驱动的安全框架 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.16088v2) [paper-pdf](https://arxiv.org/pdf/2511.16088v2)

**Authors**: Vu Van Than

**Abstract**: Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future.

摘要: 传统的威胁建模仍然以已知的TTP和过去的事件数据为中心，而威胁预测和预测框架通常与操作或架构工件脱节。这造成了一个根本性的弱点：最严重的网络威胁往往不是来自已知的，而是来自假设、忽视或尚未构想的，并且通常起源于未来，例如人工智能、信息战和供应链攻击，对手不断开发新的漏洞，可以绕过基于当前知识的防御。为了解决这一心理差距，本文介绍了未来反向威胁建模（FBTM）的理论和方法论。这种预测方法从设想的未来威胁状态开始，并向后工作以识别当前防御架构中的假设、差距、盲点和漏洞，为即将发生的威胁提供更清晰、更准确的视图，以便我们能够预测它们的出现并塑造我们想要的未来通过现在采取的行动。拟议的方法论进一步旨在揭示已知的未知数和未知的未知数，包括正在出现的、预期的和合理的策略、技术和程序。这增强了对手行为的可预测性，特别是在未来不确定性的情况下，帮助安全领导者今天做出明智的决定，为未来塑造更具弹性的安全姿态。



## **29. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness**

量子机器学习对抗鲁棒性的批判性评估 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.14989v2) [paper-pdf](https://arxiv.org/pdf/2511.14989v2)

**Authors**: Saeefa Rubaiyet Nowmi, Jesus Lopez, Md Mahmudul Alam Imon, Shahrooz Pouryousef, Mohammad Saidur Rahman

**Abstract**: Quantum Machine Learning (QML) integrates quantum computational principles into learning algorithms, offering improved representational capacity and computational efficiency. Nevertheless, the security and robustness of QML systems remain underexplored, especially under adversarial conditions. In this paper, we present a systematization of adversarial robustness in QML, integrating conceptual organization with empirical evaluation across three threat models-black-box, gray-box, and white-box. We implement representative attacks in each category, including label-flipping for black-box, QUID encoder-level data poisoning for gray-box, and FGSM and PGD for white-box, using Quantum Neural Networks (QNNs) trained on two datasets from distinct domains: MNIST from computer vision and AZ-Class from Android malware, across multiple circuit depths (2, 5, 10, and 50 layers) and two encoding schemes (angle and amplitude). Our evaluation shows that amplitude encoding yields the highest clean accuracy (93% on MNIST and 67% on AZ-Class) in deep, noiseless circuits; however, it degrades sharply under adversarial perturbations and depolarization noise (p=0.01), dropping accuracy below 5%. In contrast, angle encoding, while offering lower representational capacity, remains more stable in shallow, noisy regimes, revealing a trade-off between capacity and robustness. Moreover, the QUID attack attains higher attack success rates, though quantum noise channels disrupt the Hilbert-space correlations it exploits, weakening its impact in image domains. This suggests that noise can act as a natural defense mechanism in Noisy Intermediate-Scale Quantum (NISQ) systems. Overall, our findings guide the development of secure and resilient QML architectures for practical deployment. These insights underscore the importance of designing threat-aware models that remain reliable under real-world noise in NISQ settings.

摘要: 量子机器学习（QML）将量子计算原理集成到学习算法中，提供改进的表示能力和计算效率。然而，QML系统的安全性和稳健性仍然没有得到充分的研究，尤其是在敌对条件下。在本文中，我们提出了QML中对抗鲁棒性的系统化，将概念组织与三种威胁模型（黑箱、灰箱和白箱）的经验评估集成在一起。我们使用在来自不同领域的两个数据集上训练的量子神经网络（QNN），在每个类别中实施代表性攻击，包括针对黑匣子的标签翻转、针对灰箱的QUID编码器级数据中毒以及针对白箱的FGSM和PGP：来自计算机视觉的MNIST和来自Android恶意软件的AZ-Class，跨越多个电路深度（2、5、10和50层）和两种编码方案（角度和幅度）。我们的评估表明，幅度编码在深度无噪电路中产生最高的清晰准确性（MNIST为93%，AZ-Class为67%）;然而，在对抗性扰动和去极化噪音下，它会急剧下降（p=0.01），准确性下降到5%以下。相比之下，角度编码虽然提供较低的代表容量，但在浅层、有噪音的区域中仍然更加稳定，揭示了容量和鲁棒性之间的权衡。此外，QUID攻击获得了更高的攻击成功率，尽管量子噪音通道扰乱了它所利用的Hilbert空间相关性，削弱了它在图像域中的影响。这表明噪音可以充当有噪的中规模量子（NISQ）系统中的自然防御机制。总体而言，我们的研究结果指导了安全且有弹性的QML架构的开发以进行实际部署。这些见解强调了设计威胁感知模型的重要性，这些模型在NISQ环境中在现实世界噪音下保持可靠。



## **30. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

全规模作弊立体匹配：自动驾驶中针对双眼深度估计的物理对抗攻击 cs.CV

AAAI 2026

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.14386v3) [paper-pdf](https://arxiv.org/pdf/2511.14386v3)

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

摘要: 尽管用于实现自动驾驶感知的深度神经模型已被证明容易受到对抗示例的影响，但已知的攻击通常利用2D补丁并主要针对单目感知。因此，物理对抗示例（PAEs）对基于立体声的双眼深度估计的有效性在很大程度上仍然没有被探索。为此，我们在自动驾驶背景下提出了针对立体匹配模型的第一个基于纹理的物理对抗攻击。我们的方法采用具有全局伪装纹理的3D PCE，而不是基于局部2D补丁的纹理，确保立体相机不同视角的视觉一致性和攻击有效性。为了应对这些摄像机的差异效应，我们还提出了一种新的3D立体匹配渲染模块，该模块允许PCE与双眼视觉中的真实位置和航向对齐。我们进一步提出了一种新颖的合并攻击，通过细粒度的PCE优化将目标无缝地混合到环境中。它显着增强了对现有无法无缝合并到后台的隐藏攻击的隐蔽性和杀伤力。广泛的评估表明，我们的PEN可以成功地欺骗立体模型产生错误的深度信息。



## **31. Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion**

NLP中的隐写后门攻击：超低中毒和防御逃避 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.14301v2) [paper-pdf](https://arxiv.org/pdf/2511.14301v2)

**Authors**: Eric Xue, Ruiyi Zhang, Zijun Zhang, Pengtao Xie

**Abstract**: Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.

摘要: Transformer模型是自然语言处理（NLP）应用程序的基础，但仍然容易受到通过有毒数据引入的后门攻击，这些攻击在训练期间植入隐藏行为。为了加强防止此类妥协的能力，最近的研究重点是设计越来越隐蔽的攻击来压力测试现有的防御，将后门行为与风格化的文物或代币级扰动触发器配对。然而，这种趋势转移了人们对更困难、更现实情况的注意力：让模型响应特定名称或实体等语义触发器，其中成功的后门可以操纵与部署的系统中的真实人或事件相关的输出。受这种日益严重的脱节的激励，我们引入了SteganoBackdoor，使隐形技术与实际威胁模型重新保持一致。利用自然语言隐写术的无害属性，SteganoBackdoor应用梯度引导的数据优化过程，将语义触发种子转换为隐写载体，嵌入高后门有效载荷，保持流畅，并且没有表现出与触发器的代表性相似。在不同的实验环境中，SteganoBackdoor以比以前方法低一个数量级的数据中毒率实现了超过99%的攻击成功率，同时保持了对一套全面的数据级防御的无与伦比的规避。通过揭示这种实际和隐蔽的攻击，SteganoBackdoor强调了当前防御中的一个紧迫盲点，并要求立即关注对抗性数据防御和现实世界的威胁建模。



## **32. Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data**

动态隐私：移动传感器数据的预测性对抗转换网络 cs.CR

accepted by AAAI 2026 (oral)

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.07242v4) [paper-pdf](https://arxiv.org/pdf/2511.07242v4)

**Authors**: Tianle Song, Chenhao Lin, Yang Cao, Zhengyu Zhao, Jiahao Sun, Chong Zhang, Le Yang, Chao Shen

**Abstract**: Mobile motion sensors such as accelerometers and gyroscopes are now ubiquitously accessible by third-party apps via standard APIs. While enabling rich functionalities like activity recognition and step counting, this openness has also enabled unregulated inference of sensitive user traits, such as gender, age, and even identity, without user consent. Existing privacy-preserving techniques, such as GAN-based obfuscation or differential privacy, typically require access to the full input sequence, introducing latency that is incompatible with real-time scenarios. Worse, they tend to distort temporal and semantic patterns, degrading the utility of the data for benign tasks like activity recognition. To address these limitations, we propose the Predictive Adversarial Transformation Network (PATN), a real-time privacy-preserving framework that leverages historical signals to generate adversarial perturbations proactively. The perturbations are applied immediately upon data acquisition, enabling continuous protection without disrupting application functionality. Experiments on two datasets demonstrate that PATN substantially degrades the performance of privacy inference models, achieving Attack Success Rate (ASR) of 40.11% and 44.65% (reducing inference accuracy to near-random) and increasing the Equal Error Rate (EER) from 8.30% and 7.56% to 41.65% and 46.22%. On ASR, PATN outperforms baseline methods by 16.16% and 31.96%, respectively.

摘要: 现在，第三方应用程序通过标准API无处不在地访问加速度计和陀螺仪等移动运动传感器。这种开放性在实现活动识别和步数等丰富功能的同时，还可以在未经用户同意的情况下对敏感的用户特征（例如性别、年龄甚至身份）进行不受监管的推断。现有的隐私保护技术，例如基于GAN的模糊或差异隐私，通常需要访问完整的输入序列，从而引入与实时场景不兼容的延迟。更糟糕的是，它们往往会扭曲时间和语义模式，降低数据对活动识别等良性任务的实用性。为了解决这些限制，我们提出了预测对抗转换网络（PATN），这是一个实时隐私保护框架，可以利用历史信号主动生成对抗性扰动。数据采集后立即应用扰动，从而实现持续保护，而不会中断应用程序功能。对两个数据集的实验表明，PATN大幅降低了隐私推理模型的性能，实现了40.11%和44.65%的攻击成功率（ASB）（将推理准确率降低到接近随机），并将等错误率（EER）从8.30%和7.56%提高到41.65%和46.22%。在ASB方面，PATN分别比基线方法高出16.16%和31.96%。



## **33. CGCE: Classifier-Guided Concept Erasure in Generative Models**

CGCE：生成模型中分类器引导的概念擦除 cs.CV

26 pages, 17 figures

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.05865v2) [paper-pdf](https://arxiv.org/pdf/2511.05865v2)

**Authors**: Viet Nguyen, Vishal M. Patel

**Abstract**: Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI.

摘要: 大规模生成模型的最新进展使人们能够创建高质量的图像和视频，但也引发了有关不安全内容生成的重大安全问题。为了缓解这种情况，人们开发了概念擦除方法来从预训练的模型中删除不需要的概念。然而，现有的方法仍然容易受到可以重新生成已删除内容的对抗攻击。此外，实现稳健擦除通常会降低模型对于安全、不相关概念的生成质量，从而在安全性和性能之间产生困难的权衡。为了应对这一挑战，我们引入了分类器引导概念擦除（CGCE），这是一个高效的即插即用框架，可以为各种生成模型提供稳健的概念擦除，而不改变其原始权重。CGCE使用对文本嵌入进行操作的轻量级分类器来首先检测并细化包含不需要的概念的提示。这种方法具有高度可扩展性，允许通过汇总来自多个分类器的指导来进行多概念擦除。通过在推理时仅修改不安全的嵌入，我们的方法可以防止有害内容生成，同时在良性提示下保留模型的原始质量。大量实验表明，CGCE针对广泛的红色团队攻击实现了最先进的鲁棒性。我们的方法还保持了高的生成效用，展示了安全性和性能之间的卓越平衡。我们通过成功应用于各种现代T2 I和T2 V模型，展示了CGCE的多功能性，使其成为安全生成人工智能的实用有效解决方案。



## **34. Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks**

不一致时间：大型语言模型对对抗性攻击的鲁棒性的生存分析 cs.CL

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2510.02712v2) [paper-pdf](https://arxiv.org/pdf/2510.02712v2)

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.

摘要: 大型语言模型（LLM）彻底改变了对话人工智能，但对其在扩展多轮对话中的稳健性仍然知之甚少。现有的评估框架专注于静态基准和单轮评估，未能捕捉反映现实世界互动特征的对话退化的时间动态。在这项工作中，我们提出了对话稳健性的大规模生存分析，将故障建模为MT-Consistency基准上的9个最先进的LLM在36，951个回合内的事件时间过程。我们的框架将Cox比例风险、加速故障时间（AFT）和随机生存森林模型与简单的语义漂移特征相结合。我们发现，突然的从预定到提示的语义漂移会急剧增加不一致的风险，而累积漂移是反直觉的\{保护性}，这表明在经历多次转变的对话中进行适应。具有模型-漂移相互作用的AFT模型实现了区分和校准的最佳组合，比例风险检查揭示了关键漂移协变量的系统性违规，解释了这种环境下Cox式建模的局限性。最后，我们表明，轻量级的AFT模型可以转变为回合级风险监控器，它在第一个不一致的答案之前的几个回合标记大多数失败的对话，同时保持虚假警报适度。这些结果将生存分析确立为评估多轮稳健性和为对话式人工智能系统设计实用保障措施的强大范式。



## **35. Memory Self-Regeneration: Uncovering Hidden Knowledge in Unlearned Models**

记忆自我再生：在未学习的模型中发现隐藏的知识 cs.LG

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2510.03263v2) [paper-pdf](https://arxiv.org/pdf/2510.03263v2)

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Joanna Waczyńska, Piotr Borycki, Przemysław Spurek

**Abstract**: The impressive capability of modern text-to-image models to generate realistic visuals has come with a serious drawback: they can be misused to create harmful, deceptive or unlawful content. This has accelerated the push for machine unlearning. This new field seeks to selectively remove specific knowledge from a model's training data without causing a drop in its overall performance. However, it turns out that actually forgetting a given concept is an extremely difficult task. Models exposed to attacks using adversarial prompts show the ability to generate so-called unlearned concepts, which can be not only harmful but also illegal. In this paper, we present considerations regarding the ability of models to forget and recall knowledge, introducing the Memory Self-Regeneration task. Furthermore, we present MemoRa strategy, which we consider to be a regenerative approach supporting the effective recovery of previously lost knowledge. Moreover, we propose that robustness in knowledge retrieval is a crucial yet underexplored evaluation measure for developing more robust and effective unlearning techniques. Finally, we demonstrate that forgetting occurs in two distinct ways: short-term, where concepts can be quickly recalled, and long-term, where recovery is more challenging. Code is available at https://gmum.github.io/MemoRa/.

摘要: 现代文本到图像模型生成逼真视觉效果的令人印象深刻的能力却存在一个严重的缺点：它们可能会被滥用来创建有害、欺骗性或非法内容。这加速了机器学习的推动。这个新领域旨在从模型的训练数据中选择性地删除特定知识，而不会导致其整体性能下降。然而，事实证明，真正忘记一个给定的概念是一项极其困难的任务。暴露于使用对抗性提示的攻击的模型显示出生成所谓的未学习概念的能力，这不仅是有害的，而且是非法的。在本文中，我们提出的考虑模型的能力，忘记和回忆的知识，介绍记忆自我再生任务。此外，我们提出了MemoRa策略，我们认为这是一种再生方法，支持有效恢复以前丢失的知识。此外，我们提出，知识检索的鲁棒性是一个重要的，但未充分探索的评价措施，开发更强大和有效的unlearning技术。最后，我们证明遗忘以两种不同的方式发生：短期，概念可以被快速回忆起来，长期，恢复更具挑战性。代码可在https://gmum.github.io/MemoRa/上获取。



## **36. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意你的服务器：MCP生态系统上寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2509.06572v2) [paper-pdf](https://arxiv.org/pdf/2509.06572v2)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **37. Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics**

发现和缓解Deepfake主动取证中的破坏性多重嵌入攻击 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2508.17247v2) [paper-pdf](https://arxiv.org/pdf/2508.17247v2)

**Authors**: Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang

**Abstract**: With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.

摘要: 随着Deepfake技术的快速发展和数字媒体的广泛传播，个人隐私面临着日益严重的安全威胁。Deepfake主动取证涉及嵌入不可感知的水印以实现可靠的源跟踪，是抵御这些威胁的重要防御措施。尽管现有的方法显示出很强的取证能力，但它们依赖于单个水印嵌入的理想化假设，这在现实世界的场景中被证明是不切实际的。本文首次正式定义并证明了多重嵌入攻击（MEA）的存在性。当之前保护的图像经历额外几轮水印嵌入时，原始的取证水印可能会被破坏或删除，从而导致整个主动取证机制无效。为了解决这个漏洞，我们提出了一种名为对抗干扰模拟（AIS）的通用训练范式。AIS没有修改网络架构，而是在微调期间显式地模拟了多边环境协议（MTA）场景，并引入了顺从驱动的损失函数来强制学习稀疏和稳定的水印表示。我们的方法使模型即使在第二次嵌入之后也能够保持正确提取原始水印的能力。大量实验表明，我们的即插即用的AIS训练范式显着增强了各种现有方法针对MTA的鲁棒性。



## **38. Special-Character Adversarial Attacks on Open-Source Language Model**

开源语言模型的特殊字符对抗攻击 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2508.14070v2) [paper-pdf](https://arxiv.org/pdf/2508.14070v2)

**Authors**: Ephraiem Sarabamoun

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. This paper presents a study of different special character attacks including unicode, homoglyph, structural, and textual encoding attacks aimed at bypassing safety mechanisms. We evaluate seven prominent open-source models ranging from 3.8B to 32B parameters on 4,000+ attack attempts. These experiments reveal critical vulnerabilities across all model sizes, exposing failure modes that include successful jailbreaks, incoherent outputs, and unrelated hallucinations.

摘要: 大型语言模型（LLM）在各种自然语言处理任务中取得了非凡的性能，但它们对字符级对抗性操纵的脆弱性给现实世界的部署带来了重大的安全挑战。本文研究了旨在绕过安全机制的不同特殊字符攻击，包括Unicode、同字形、结构性和文本编码攻击。我们对4，000多次攻击尝试进行了评估，参数范围从3.8B到32 B不等。这些实验揭示了所有模型尺寸的关键漏洞，揭示了包括成功越狱、不连贯的输出和不相关的幻觉在内的失败模式。



## **39. Large Language Model Unlearning for Source Code**

大型语言模型放弃源代码的学习 cs.SE

Accepted to AAAI'26

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2506.17125v2) [paper-pdf](https://arxiv.org/pdf/2506.17125v2)

**Authors**: Xue Jiang, Yihong Dong, Huangzhao Zhang, Tangxinyu Wang, Zheng Fang, Yingwei Ma, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Yongbin Li, Ge Li

**Abstract**: While Large Language Models (LLMs) excel at code generation, their inherent tendency toward verbatim memorization of training data introduces critical risks like copyright infringement, insecure emission, and deprecated API utilization, etc. A straightforward yet promising defense is unlearning, ie., erasing or down-weighting the offending snippets through post-training. However, we find its application to source code often tends to spill over, damaging the basic knowledge of programming languages learned by the LLM and degrading the overall capability. To ease this challenge, we propose PROD for precise source code unlearning. PROD surgically zeroes out the prediction probability of the prohibited tokens, and renormalizes the remaining distribution so that the generated code stays correct. By excising only the targeted snippets, PROD achieves precise forgetting without much degradation of the LLM's overall capability. To facilitate in-depth evaluation against PROD, we establish an unlearning benchmark consisting of three downstream tasks (ie., unlearning of copyrighted code, insecure code, and deprecated APIs), and introduce Pareto Dominance Ratio (PDR) metric, which indicates both the forget quality and the LLM utility. Our comprehensive evaluation demonstrates that PROD achieves superior overall performance between forget quality and model utility compared to existing unlearning approaches across three downstream tasks, while consistently exhibiting improvements when applied to LLMs of varying series. PROD also exhibits superior robustness against adversarial attacks without generating or exposing the data to be forgotten. These results underscore that our approach not only successfully extends the application boundary of unlearning techniques to source code, but also holds significant implications for advancing reliable code generation.

摘要: 虽然大型语言模型（LLM）擅长代码生成，但其固有的逐字记忆训练数据的倾向会引入版权侵权、不安全的发射和过时的API利用等关键风险。一个简单但有希望的防御是取消学习，即，通过训练后删除或降低违规片段的权重。然而，我们发现它对源代码的应用往往会溢出，损害LLM学到的编程语言的基本知识，并降低整体能力。为了缓解这一挑战，我们提出了PROD来精确的源代码反学习。PROD通过外科手术将被禁止的令牌的预测概率归零，并重新规范剩余的分布，以便生成的代码保持正确。通过仅删除目标片段，PROD实现了精确遗忘，而不会大幅降低LLM的整体能力。为了促进针对PROD的深入评估，我们建立了一个由三个下游任务（即，放弃受版权保护的代码、不安全的代码和废弃的API），并引入帕累托主导比（PDR）指标，该指标既指示忘记质量又指示LLM实用性。我们的全面评估表明，与三个下游任务中的现有取消学习方法相比，PROD在忘记质量和模型效用之间实现了更好的整体性能，同时在应用于不同系列的LLM时一致表现出改进。PROD还表现出针对对抗攻击的卓越鲁棒性，而不会生成或暴露被遗忘的数据。这些结果强调，我们的方法不仅成功地将放弃学习技术的应用边界扩展到源代码，而且对推进可靠的代码生成具有重要影响。



## **40. TRAP: Targeted Redirecting of Agentic Preferences**

TRAP：有针对性地重新定向统计偏好 cs.AI

Accepted to NeurIPS 2025

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2505.23518v2) [paper-pdf](https://arxiv.org/pdf/2505.23518v2)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a novel generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections into the vision-language embedding space. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP consistently induces decision-level preference redirection on leading models, including LLaVA-34B, Gemma3, GPT-4o, and Mistral-3.2, significantly outperforming existing baselines such as SPSA, Bandit, and standard diffusion approaches. These findings expose a critical, generalized vulnerability: autonomous agents can be consistently misled through visually subtle, semantically-guided cross-modal manipulations. Overall, our results show the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making. The code for TRAP is accessible on GitHub at https://github.com/uiuc-focal-lab/TRAP.

摘要: 由视觉语言模型（VLM）提供支持的自主代理人工智能系统正在迅速向现实世界的部署迈进，但它们的跨模式推理能力为对抗性操纵引入了新的攻击表面，利用跨模式的语义推理。现有的对抗性攻击通常依赖于可见像素扰动或需要特权模型或环境访问，这使得它们对于隐形的、现实世界的利用来说不切实际。我们引入TRAP，这是一种新型的生成对抗框架，它使用基于扩散的语义注入到视觉语言嵌入空间来操纵代理的决策。我们的方法在连体语义网络和布局感知空间掩蔽的指导下，将基于预算的负降级与正语义优化相结合。在不需要访问模型内部的情况下，TRAP会产生视觉上自然的图像，但在代理人工智能系统中会引起一致的选择偏差。我们在Microsoft上下文中的公共对象（COCO）数据集中评估TRAP，构建多候选决策场景。在这些场景中，TRAP一致地在LLaVA-34 B、Gemma 3、GPT-4 o和Mistral-3.2等领先模型上引发决策级偏好重定向，显着优于SPSA、Bandit和标准扩散方法等现有基线。这些发现暴露了一个关键的、普遍的弱点：自主代理可能会通过视觉上微妙的、语义引导的跨模式操纵持续误导。总体而言，我们的结果表明，需要像素级稳健性之外的防御策略来解决跨模式决策中的语义漏洞。TRAP的代码可在GitHub上访问：https://github.com/uiuc-focal-lab/TRAP。



## **41. Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration**

通过高效校准实现统一多模式编码器的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2505.11895v2) [paper-pdf](https://arxiv.org/pdf/2505.11895v2)

**Authors**: Chih-Ting Liao, Zhangquan Chen, Chunlei Meng, Tzu-Yu Huang, Xin Cao, Xu Zheng

**Abstract**: Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.

摘要: 最近的统一多模态编码器将各种模态对齐到共享表示空间中，从而实现各种跨模态任务。尽管这些模型具有令人印象深刻的能力，但它们在对抗性扰动下的鲁棒性仍然没有得到充分的研究，这对于安全敏感的应用程序来说是一个关键问题。在这项工作中，我们首次全面研究了统一多模态编码器中的对抗脆弱性。我们发现，即使是轻微的对抗性扰动也会导致所有模式的性能大幅下降。音频和点云等非视觉输入尤其脆弱，而图像和视频等视觉输入也会显着退化。为了解决这个问题，我们提出了一种高效的对抗性校准框架，该框架在无需修改预训练的编码器或语义中心的情况下提高了各个模式的鲁棒性，确保与现有基础模型的兼容性。我们的方法引入了仅在对抗性示例上训练的特定模式投影头，同时保持主干和嵌入冻结。我们探索了三个训练目标：固定中心交叉熵、干净对抗L2对齐和干净对抗InfoNSO，并引入了一种正规化策略来确保在攻击下的模式一致对齐。对六种模式和三种绑定风格模型的实验表明，我们的方法在RST = 4/255时将对抗鲁棒性提高了高达47.3%，同时保留甚至改进了干净的零射击和检索性能，可训练参数少于1%。



## **42. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

通过自然和对抗局部破坏对DNN的空间鲁棒性进行基准测试 cs.CV

Accepted for publication in Pattern Recognition

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2504.01632v3) [paper-pdf](https://arxiv.org/pdf/2504.01632v3)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of deep neural networks is a crucial factor in safety-critical applications, particularly in complex and dynamic environments (e.g., medical or driving scenarios) where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remains underexplored. This paper fills this gap by introducing novel, region-aware metrics for benchmarking the spatial robustness of segmentation models, along with an evaluation framework to assess the impact of natural localized corruptions. Furthermore, it uncovers the inherent complexity of evaluating worst-case spatial robustness using only a single localized adversarial attack. To address this, the work proposes a region-aware multi-attack adversarial analysis to systematically assess model robustness across specific image regions. The proposed metrics and analysis were exploited to evaluate 14 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones, and vice versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.

摘要: 深度神经网络的稳健性是安全关键应用中的一个关键因素，特别是在复杂和动态的环境中（例如，医疗或驾驶场景），其中可能会出现局部腐败。虽然之前的研究已经评估了语义分割（SS）模型在全图像自然或对抗破坏下的鲁棒性，但对局部破坏下密集视觉模型的空间鲁棒性的全面研究仍然不足。本文通过引入新颖的区域感知指标来对分割模型的空间稳健性进行基准测试，以及评估自然局部破坏的影响的评估框架来填补这一空白。此外，它揭示了仅使用单个局部对抗攻击来评估最坏情况空间稳健性的固有复杂性。为了解决这个问题，该工作提出了一种区域感知的多攻击对抗分析，以系统性地评估特定图像区域的模型稳健性。利用提出的指标和分析来评估驾驶场景中的14个细分模型，揭示了对自然和对抗形式的局部腐败影响的关键见解。结果表明，模型对这两种类型的威胁的反应不同;例如，基于变换器的分割模型对局部自然破坏表现出显着的鲁棒性，但极易受到对抗性破坏的影响，而基于CNN的模型反之亦然。因此，我们还通过集成模型解决了平衡对自然和对抗局部破坏的鲁棒性的挑战，从而实现更广泛的威胁覆盖范围并提高密集视觉任务的可靠性。



## **43. ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense**

ARBoids：采用Boids模型的自适应剩余强化学习用于协作多USV目标防御 cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2502.18549v3) [paper-pdf](https://arxiv.org/pdf/2502.18549v3)

**Authors**: Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang

**Abstract**: The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.

摘要: 无人水面航行器（USV）的目标防御问题（SDP）涉及在敌方USV突破指定目标区域之前使用一辆或多辆防御USV拦截其。当攻击者与防御者相比表现出更好的机动性时，就会出现一种特别具有挑战性的情况，从而使有效拦截变得非常复杂。为了应对这一挑战，这封信介绍了ARBoids，这是一种新型的自适应剩余强化学习框架，它将深度强化学习（DRL）与生物启发的、基于力的Boids模型集成在一起。在此框架中，Boids模型充当多智能体协调的计算高效基线策略，而DRL则学习剩余策略以自适应地细化和优化防御者的动作。所提出的方法在高保真Gazebo模拟环境中得到了验证，证明了优于传统拦截策略（包括纯粹基于力量的方法和普通DRL策略）的性能。此外，学习到的策略对具有不同机动性特征的攻击者表现出很强的适应性，凸显了其鲁棒性和概括能力。ARBoids的代码将在接受本信函后发布。



## **44. DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs**

DarkMind：定制LLC中潜在的思想链后门 cs.CR

19 pages, 15 figures, 12 tables

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18617v2) [paper-pdf](https://arxiv.org/pdf/2501.18617v2)

**Authors**: Zhen Guo, Shanghao Shi, Shamim Yazdani, Ning Zhang, Reza Tourani

**Abstract**: With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.

摘要: 随着个性化人工智能的迅速崛起，配备思想链（COT）推理的定制大型语言模型（LLM）现在为数百万人工智能代理提供动力。然而，它们复杂的推理过程会引入新的且基本上未被探索的安全漏洞。我们提出了DarkMind，这是一种新颖的潜在推理级后门攻击，通过在不改变用户查询的情况下操纵内部COT步骤来针对自定义的LLM。与之前的基于提示的攻击不同，DarkMind通过潜在触发器在推理链中秘密激活，无需修改输入提示或要求访问模型参数即可实现对抗行为。为了实现隐形和可靠性，我们提出了即时和追溯双重触发类型，并将它们集成到统一的嵌入模板中，该模板管理触发相关激活，采用隐形优化算法来最大限度地减少语义漂移，并引入自动对话启动器跨域的秘密激活。使用五个LLM对跨越算术、常识和符号领域的八个推理数据集进行了全面实验，证明DarkMind始终实现了高攻击成功率。我们进一步研究了减轻这些风险的防御策略，并揭示了推理级后门代表了一个重大但未充分探索的威胁，强调了对强大、推理感知安全机制的必要性。



## **45. On the Effectiveness of Adversarial Training on Malware Classifiers**

恶意软件分类器对抗训练的有效性 cs.LG

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2412.18218v2) [paper-pdf](https://arxiv.org/pdf/2412.18218v2)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) is a key defense against Machine Learning evasion attacks, but its effectiveness for real-world malware detection remains poorly understood. This uncertainty stems from a critical disconnect in prior research: studies often overlook the inherent nature of malware and are fragmented, examining diverse variables like realism or confidence of adversarial examples in isolation, or relying on weak evaluations that yield non-generalizable insights. To address this, we introduce Rubik, a framework for the systematic, multi-dimensional evaluation of AT in the malware domain. This framework defines diverse key factors across essential dimensions, including data, feature representations, classifiers, and robust optimization settings, for a comprehensive exploration of the interplay of influential AT's variables through reliable evaluation practices, such as realistic evasion attacks. We instantiate Rubik on Android malware, empirically analyzing how this interplay shapes robustness. Our findings challenge prior beliefs--showing, for instance, that realizable adversarial examples offer only conditional robustness benefits--and reveal new insights, such as the critical role of model architecture and feature-space structure in determining AT's success. From this analysis, we distill four key insights, expose four common evaluation misconceptions, and offer practical recommendations to guide the development of truly robust malware classifiers.

摘要: 对抗训练（AT）是抵御机器学习规避攻击的关键防御措施，但其对现实世界恶意软件检测的有效性仍然知之甚少。这种不确定性源于之前研究中的严重脱节：研究经常忽视恶意软件的固有本质，并且是碎片化的，孤立地检查各种变量，例如现实主义或对抗性示例的信心，或者依赖于产生不可概括见解的薄弱评估。为了解决这个问题，我们引入了Rubik，这是一个用于在恶意软件领域对AT进行系统性、多维评估的框架。该框架定义了各个基本维度的不同关键因素，包括数据、特征表示、分类器和稳健的优化设置，以便通过可靠的评估实践（例如现实的规避攻击）全面探索有影响力的AT变量的相互作用。我们在Android恶意软件上实例化Rubik，以经验分析这种相互作用如何塑造稳健性。我们的研究结果挑战了先前的信念--例如，表明可实现的对抗性示例仅提供条件稳健性好处--并揭示了新的见解，例如模型架构和特征空间结构在决定AT成功方面的关键作用。从此分析中，我们提炼出四个关键见解，揭露四个常见的评估误解，并提供实用建议来指导真正强大的恶意软件分类器的开发。



## **46. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2410.15236v3) [paper-pdf](https://arxiv.org/pdf/2410.15236v3)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin, Xinyuan Song

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型（LLM）通过推进自然语言理解和生成来改变了人工智能，实现了医疗保健、软件工程和对话系统以外的各个领域的应用。尽管过去几年取得了这些进步，但LLM仍表现出相当大的漏洞，特别是在引发注射和越狱攻击方面。本评论分析了这些漏洞的研究状况，并提出了可用的防御策略。我们将攻击方法大致分为基于预算、基于模型、多模式和多语言，涵盖对抗提示、后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括即时过滤、转换、对齐技术、多智能体防御和自我调节，评估它们的优点和缺点。我们还讨论了用于评估LLM安全性和稳健性的关键指标和基准，并指出了交互式环境中攻击成功的量化以及现有数据集中的偏差等挑战。通过识别当前的研究差距，我们提出了弹性对齐策略、针对不断发展的攻击的先进防御、越狱检测自动化以及道德和社会影响的未来方向。该审查强调了人工智能社区内持续研究与合作的必要性，以增强LLM安全性并确保其安全部署。



## **47. A Gray-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

对基于潜在扩散模型的图像编辑的灰箱攻击 cs.CV

15 pages, 9 figures, 9 tables

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2408.10901v4) [paper-pdf](https://arxiv.org/pdf/2408.10901v4)

**Authors**: Zhongliang Guo, Chun Tong Lei, Lei Fang, Shuai Zhao, Yifei Qian, Jingyu Lin, Zeyu Wang, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in Latent Diffusion Models (LDMs) have revolutionized image synthesis and manipulation, raising significant concerns about data misappropriation and intellectual property infringement. While adversarial attacks have been extensively explored as a protective measure against such misuse of generative AI, current approaches are severely limited by their heavy reliance on model-specific knowledge and substantial computational costs. Drawing inspiration from the posterior collapse phenomenon observed in VAE training, we propose the Posterior Collapse Attack (PCA), a novel framework for protecting images from unauthorized manipulation. Through comprehensive theoretical analysis and empirical validation, we identify two distinct collapse phenomena during VAE inference: diffusion collapse and concentration collapse. Based on this discovery, we design a unified loss function that can flexibly achieve both types of collapse through parameter adjustment, each corresponding to different protection objectives in preventing image manipulation. Our method significantly reduces dependence on model-specific knowledge by requiring access to only the VAE encoder, which constitutes less than 4\% of LDM parameters. Notably, PCA achieves prompt-invariant protection by operating on the VAE encoder before text conditioning occurs, eliminating the need for empty prompt optimization required by existing methods. This minimal requirement enables PCA to maintain adequate transferability across various VAE-based LDM architectures while effectively preventing unauthorized image editing. Extensive experiments show PCA outperforms existing techniques in protection effectiveness, computational efficiency (runtime and VRAM), and generalization across VAE-based LDM variants. Our code is available at https://github.com/ZhongliangGuo/PosteriorCollapseAttack.

摘要: 潜在扩散模型（LDM）的最新进展彻底改变了图像合成和操纵，引发了人们对数据盗用和知识产权侵犯的严重担忧。虽然对抗攻击已被广泛探索，作为防止此类滥用生成性人工智能的保护措施，但当前的方法因严重依赖特定模型的知识和高昂的计算成本而受到严重限制。我们从VAE训练中观察到的后部塌陷现象中汲取灵感，提出了后部塌陷攻击（PCA），这是一种用于保护图像免受未经授权操纵的新型框架。通过全面的理论分析和实证验证，我们发现了VAE推断过程中两种不同的崩溃现象：扩散崩溃和浓度崩溃。基于这一发现，我们设计了一个统一的损失函数，可以通过参数调整灵活实现两种类型的崩溃，每种都对应于防止图像操纵的不同保护目标。我们的方法通过仅要求访问VAE编码器（其构成的LDM参数不到4%），显着减少了对模型特定知识的依赖。值得注意的是，PCA通过在文本条件处理发生之前对VAE编码器进行操作来实现预算不变保护，从而消除了现有方法所需的空提示优化的需要。这一最低要求使PCA能够在各种基于VAE的LDM架构中保持足够的可移植性，同时有效地防止未经授权的图像编辑。大量实验表明，PCA在保护有效性、计算效率（运行时和VRAM）以及基于VAE的LDM变体的概括方面优于现有技术。我们的代码可以在https://github.com/ZhongliangGuo/PosteriorCollapseAttack上找到。



## **48. HO-FMN: Hyperparameter Optimization for Fast Minimum-Norm Attacks**

HO-FNN：针对快速最小规范攻击的超参数优化 cs.LG

Accepted at Neurocomputing

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2407.08806v3) [paper-pdf](https://arxiv.org/pdf/2407.08806v3)

**Authors**: Raffaele Mura, Giuseppe Floris, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Giorgio Giacinto, Battista Biggio, Fabio Roli

**Abstract**: Gradient-based attacks are a primary tool to evaluate robustness of machine-learning models. However, many attacks tend to provide overly-optimistic evaluations as they use fixed loss functions, optimizers, step-size schedulers, and default hyperparameters. In this work, we tackle these limitations by proposing a parametric variation of the well-known fast minimum-norm attack algorithm, whose loss, optimizer, step-size scheduler, and hyperparameters can be dynamically adjusted. We re-evaluate 12 robust models, showing that our attack finds smaller adversarial perturbations without requiring any additional tuning. This also enables reporting adversarial robustness as a function of the perturbation budget, providing a more complete evaluation than that offered by fixed-budget attacks, while remaining efficient. We release our open-source code at https://github.com/pralab/HO-FMN.

摘要: 基于对象的攻击是评估机器学习模型稳健性的主要工具。然而，许多攻击往往会提供过于乐观的评估，因为它们使用固定损失函数、优化器、步进大小排序器和默认超参数。在这项工作中，我们通过提出著名的快速最小模攻击算法的参数变体来解决这些限制，该算法的损失、优化器、步进大小调度器和超参数可以动态调整。我们重新评估了12个稳健模型，表明我们的攻击可以发现更小的对抗扰动，而不需要任何额外的调整。这还使得能够将对抗稳健性报告为扰动预算的函数，从而提供比固定预算攻击更完整的评估，同时保持高效。我们在https://github.com/pralab/HO-FMN上发布我们的开源代码。



## **49. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

数据驱动的Lipschitz连续性：提高对抗稳健性的经济有效方法 cs.LG

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2406.19622v2) [paper-pdf](https://arxiv.org/pdf/2406.19622v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: As deep neural networks (DNNs) are increasingly deployed in sensitive applications, ensuring their security and robustness has become critical. A major threat to DNNs arises from adversarial attacks, where small input perturbations can lead to incorrect predictions. Recent advances in adversarial training improve robustness by incorporating additional examples from external datasets or generative models. However, these methods often incur high computational costs, limiting their practicality and hindering real-world deployment. In this paper, we propose a cost-efficient alternative based on Lipschitz continuity that achieves robustness comparable to models trained with extensive supplementary data. Unlike conventional adversarial training, our method requires only a single pass over the dataset without gradient estimation, making it highly efficient. Furthermore, our method can integrate seamlessly with existing adversarial training frameworks and enhances the robustness of models without requiring extra generative data. Experimental results show that our approach not only reduces computational overhead but also maintains or improves the defensive capabilities of robust neural networks. This work opens a promising direction for developing practical, scalable defenses against adversarial attacks.

摘要: 随着深度神经网络（DNN）越来越多地部署在敏感应用中，确保其安全性和稳健性变得至关重要。DNN的主要威胁来自对抗性攻击，其中微小的输入扰动可能会导致错误的预测。对抗训练的最新进展通过整合来自外部数据集或生成模型的额外示例来提高稳健性。然而，这些方法通常会产生很高的计算成本，限制了它们的实用性并阻碍了现实世界的部署。在本文中，我们提出了一种基于Lipschitz连续性的经济高效替代方案，其鲁棒性与使用大量补充数据训练的模型相当。与传统的对抗训练不同，我们的方法只需要在数据集上进行一次传递，而无需进行梯度估计，因此非常高效。此外，我们的方法可以与现有的对抗训练框架无缝集成，并增强模型的稳健性，而不需要额外的生成数据。实验结果表明，我们的方法不仅减少了计算负担，而且还保持或提高了鲁棒神经网络的防御能力。这项工作为开发针对对抗性攻击的实用、可扩展的防御开辟了一个有希望的方向。



## **50. LTD: Low Temperature Distillation for Gradient Masking-free Adversarial Training**

LTD：低温蒸馏用于无梯度掩蔽对抗训练 cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2111.02331v4) [paper-pdf](https://arxiv.org/pdf/2111.02331v4)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstract**: Adversarial training is a widely adopted strategy to bolster the robustness of neural network models against adversarial attacks. This paper revisits the fundamental assumptions underlying image classification and suggests that representing data as one-hot labels is a key factor that leads to vulnerabilities. However, in real-world datasets, data ambiguity often arises, with samples exhibiting characteristics of multiple classes, rendering one-hot label representations imprecise. To address this, we introduce a novel approach, Low-Temperature Distillation (LTD), designed to refine label representations. Unlike previous approaches, LTD incorporates a relatively low temperature in the teacher model, while maintaining a fixed temperature for the student model during both training and inference. This strategy not only refines assumptions about data distribution but also strengthens model robustness and avoids the gradient masking problem commonly encountered in defensive distillation. Experimental results demonstrate the efficacy of the proposed method when combined with existing frameworks, achieving robust accuracy rates of 58.19%, 31.13%, and 42.08% on the CIFAR-10, CIFAR-100, and ImageNet datasets, respectively, without the need for additional data.

摘要: 对抗训练是一种广泛采用的策略，用于增强神经网络模型对抗对抗攻击的鲁棒性。本文重新探讨了图像分类的基本假设，并建议将数据表示为单一标签是导致漏洞的关键因素。然而，在现实世界的数据集中，经常出现数据模糊性，样本表现出多个类别的特征，导致单一标签表示不精确。为了解决这个问题，我们引入了一种新颖的方法--低温蒸馏（LTD），旨在完善标签表示。与以前的方法不同，LTD在教师模型中引入了相对较低的温度，同时在训练和推理期间为学生模型保持固定的温度。该策略不仅完善了有关数据分布的假设，而且增强了模型的稳健性，并避免了防御性蒸馏中常见的梯度掩蔽问题。实验结果证明，与现有框架相结合时，所提出的方法的有效性，在CIFAR-10、CIFAR-100和ImageNet数据集上分别实现了58.19%、31.13%和42.08%的稳健准确率，无需额外数据。



