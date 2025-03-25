# Latest Adversarial Attack Papers
**update at 2025-03-25 09:59:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Model-Guardian: Protecting against Data-Free Model Stealing Using Gradient Representations and Deceptive Predictions**

模型守护者：使用梯度表示和欺骗性预测防止无数据模型窃取 cs.CR

Full version of the paper accepted by ICME 2025

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.18081v1) [paper-pdf](http://arxiv.org/pdf/2503.18081v1)

**Authors**: Yunfei Yang, Xiaojun Chen, Yuexin Xuan, Zhendong Zhao

**Abstract**: Model stealing attack is increasingly threatening the confidentiality of machine learning models deployed in the cloud. Recent studies reveal that adversaries can exploit data synthesis techniques to steal machine learning models even in scenarios devoid of real data, leading to data-free model stealing attacks. Existing defenses against such attacks suffer from limitations, including poor effectiveness, insufficient generalization ability, and low comprehensiveness. In response, this paper introduces a novel defense framework named Model-Guardian. Comprising two components, Data-Free Model Stealing Detector (DFMS-Detector) and Deceptive Predictions (DPreds), Model-Guardian is designed to address the shortcomings of current defenses with the help of the artifact properties of synthetic samples and gradient representations of samples. Extensive experiments on seven prevalent data-free model stealing attacks showcase the effectiveness and superior generalization ability of Model-Guardian, outperforming eleven defense methods and establishing a new state-of-the-art performance. Notably, this work pioneers the utilization of various GANs and diffusion models for generating highly realistic query samples in attacks, with Model-Guardian demonstrating accurate detection capabilities.

摘要: 模型窃取攻击正日益威胁部署在云中的机器学习模型的机密性。最近的研究表明，即使在缺乏真实数据的情况下，攻击者也可以利用数据合成技术窃取机器学习模型，从而导致无数据模型窃取攻击。现有的针对此类攻击的防御存在有效性差、泛化能力不足、全面性低等局限性。对此，本文提出了一种名为Model-Guardian的新型防御框架。模型卫士由无数据模型窃取检测器(DFMS-检测器)和欺骗性预测(DPreds)两部分组成，旨在利用合成样本的伪影特性和样本的梯度表示来解决现有防御措施的不足。在7种流行的无数据模型窃取攻击上的广泛实验表明了Model-Guardian的有效性和优越的泛化能力，其性能超过了11种防御方法，并建立了新的最先进的性能。值得注意的是，这项工作开创了利用各种GAN和扩散模型在攻击中生成高度逼真的查询样本的先河，Model-Guardian展示了准确的检测能力。



## **2. Metaphor-based Jailbreaking Attacks on Text-to-Image Models**

基于隐喻的文本到图像模型越狱攻击 cs.CR

13 page3, 4 figures. This paper includes model-generated content that  may contain offensive or distressing material

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17987v1) [paper-pdf](http://arxiv.org/pdf/2503.17987v1)

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu

**Abstract**: To mitigate misuse, text-to-image~(T2I) models commonly incorporate safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods use LLMs to generate adversarial prompts that effectively bypass safety filters while generating sensitive images, revealing the safety vulnerabilities within the T2I model. However, existing LLM-based attack methods lack explicit guidance, relying on substantial queries to achieve a successful attack, which limits their practicality in real-world scenarios. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to balance the attack effectiveness and query efficiency by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance the attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Experiments demonstrate that MJA achieves better attack effectiveness while requiring fewer queries compared to baseline methods. Moreover, our adversarial prompts exhibit strong transferability across various open-source and commercial T2I models. \textcolor{red}{This paper includes model-generated content that may contain offensive or distressing material.}

摘要: 为了减少误用，文本到图像~(T2I)模型通常包含安全过滤器以防止生成敏感图像。不幸的是，最近的越狱攻击方法使用LLMS来生成敌意提示，在生成敏感图像的同时有效地绕过安全过滤器，从而暴露了T2I模型中的安全漏洞。然而，现有的基于LLM的攻击方法缺乏明确的指导，依赖大量的查询来实现成功的攻击，这限制了它们在现实场景中的实用性。在本文中，我们引入了一种受禁忌游戏启发的基于隐喻的中断方法-.具体而言，MJA由两个模块组成：基于LLM的多智能体生成模块~(MLAG)和对抗性提示优化模块~(APO)。MLAG将基于隐喻的对抗性提示生成分解为三个子任务：隐喻检索、语境匹配和对抗性提示生成。随后，MLAG协调三个基于LLM的代理通过探索各种隐喻和上下文来生成不同的对抗性提示。为了提高攻击效率，APO首先训练代理模型预测对抗性提示的攻击结果，然后设计获取策略自适应地识别最优对抗性提示。实验表明，与基准方法相比，MJA在减少查询次数的同时，取得了更好的攻击效果。此外，我们的敌意提示在各种开源和商业T2I模型之间显示出很强的可转移性。\extCOLOR{RED}{本文包括可能包含攻击性或令人不快的内容的模型生成的内容。}



## **3. STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models**

STShield：用于大型语言模型中实时越狱检测的单令牌哨兵 cs.CL

11 pages

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17932v1) [paper-pdf](http://arxiv.org/pdf/2503.17932v1)

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment.

摘要: 大型语言模型(LLM)越来越容易受到绕过其安全机制的越狱攻击。虽然现有的防御方法要么遭受自适应攻击，要么需要计算昂贵的辅助模型，我们提出了STShield，一个用于实时越狱判断的轻量级框架。STShield引入了一种新颖的单令牌哨兵机制，该机制将一个二进制安全指示器附加到模型的响应序列中，利用LLM自己的对准能力进行检测。我们的框架结合了对正常提示的监督微调和使用嵌入空间扰动的对抗性训练，在保持模型实用性的同时实现了稳健的检测。大量的实验表明，STShield成功地防御了各种越狱攻击，同时保持了该模型在合法查询上的性能。与现有方法相比，STShield以最小的计算开销实现了卓越的防御性能，使其成为现实世界LLM部署的实用解决方案。



## **4. Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation**

通过不同参数增强提高人脸识别对抗攻击的可转移性 cs.CV

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2411.15555v2) [paper-pdf](http://arxiv.org/pdf/2411.15555v2)

**Authors**: Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang

**Abstract**: Face Recognition (FR) models are vulnerable to adversarial examples that subtly manipulate benign face images, underscoring the urgent need to improve the transferability of adversarial attacks in order to expose the blind spots of these systems. Existing adversarial attack methods often overlook the potential benefits of augmenting the surrogate model with diverse initializations, which limits the transferability of the generated adversarial examples. To address this gap, we propose a novel method called Diverse Parameters Augmentation (DPA) attack method, which enhances surrogate models by incorporating diverse parameter initializations, resulting in a broader and more diverse set of surrogate models. Specifically, DPA consists of two key stages: Diverse Parameters Optimization (DPO) and Hard Model Aggregation (HMA). In the DPO stage, we initialize the parameters of the surrogate model using both pre-trained and random parameters. Subsequently, we save the models in the intermediate training process to obtain a diverse set of surrogate models. During the HMA stage, we enhance the feature maps of the diversified surrogate models by incorporating beneficial perturbations, thereby further improving the transferability. Experimental results demonstrate that our proposed attack method can effectively enhance the transferability of the crafted adversarial face examples.

摘要: 人脸识别(FR)模型很容易受到敌意例子的攻击，这些例子巧妙地操纵了良性的人脸图像，这突显了迫切需要提高对抗性攻击的可转移性，以暴露这些系统的盲点。现有的对抗性攻击方法往往忽略了用不同的初始化来扩充代理模型的潜在好处，这限制了生成的对抗性实例的可转移性。为了弥补这一缺陷，我们提出了一种新的方法，称为不同参数增强(DPA)攻击方法，它通过结合不同的参数初始化来增强代理模型，从而产生更广泛和更多样化的代理模型集。具体地说，DPA包括两个关键阶段：多参数优化(DPO)和硬模型聚合(HMA)。在DPO阶段，我们使用预先训练的参数和随机参数来初始化代理模型的参数。随后，我们在中间训练过程中保存模型，以获得多样化的代理模型集。在HMA阶段，我们通过加入有益的扰动来增强多样化代理模型的特征映射，从而进一步提高了可转移性。实验结果表明，本文提出的攻击方法可以有效地提高特制的对抗性人脸样本的可转移性。



## **5. Detecting and Mitigating DDoS Attacks with AI: A Survey**

使用人工智能检测和缓解DDOS攻击：一项调查 cs.CR

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17867v1) [paper-pdf](http://arxiv.org/pdf/2503.17867v1)

**Authors**: Alexandru Apostu, Silviu Gheorghe, Andrei Hîji, Nicolae Cleju, Andrei Pătraşcu, Cristian Rusu, Radu Ionescu, Paul Irofti

**Abstract**: Distributed Denial of Service attacks represent an active cybersecurity research problem. Recent research shifted from static rule-based defenses towards AI-based detection and mitigation. This comprehensive survey covers several key topics. Preeminently, state-of-the-art AI detection methods are discussed. An in-depth taxonomy based on manual expert hierarchies and an AI-generated dendrogram are provided, thus settling DDoS categorization ambiguities. An important discussion on available datasets follows, covering data format options and their role in training AI detection methods together with adversarial training and examples augmentation. Beyond detection, AI based mitigation techniques are surveyed as well. Finally, multiple open research directions are proposed.

摘要: 分布式拒绝服务攻击代表了一个活跃的网络安全研究问题。最近的研究从基于规则的静态防御转向基于人工智能的检测和缓解。这项全面的调查涵盖了几个关键主题。主要讨论了最先进的人工智能检测方法。提供了基于手动专家层次结构和AI生成的树图的深入分类，从而解决了DDOS分类的模糊性。以下是对可用数据集的重要讨论，涵盖数据格式选项及其在训练人工智能检测方法以及对抗性训练和示例增强中的作用。除了检测之外，还调查了基于人工智能的缓解技术。最后，提出了多个开放的研究方向。



## **6. A Causal Analysis of the Plots of Intelligent Adversaries**

智能对手情节的因果分析 stat.ME

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17863v1) [paper-pdf](http://arxiv.org/pdf/2503.17863v1)

**Authors**: Preetha Ramiah, David I. Hastie, Oliver Bunnin, Silvia Liverani, James Q. Smith

**Abstract**: In this paper we demonstrate a new advance in causal Bayesian graphical modelling combined with Adversarial Risk Analysis. This research aims to support strategic analyses of various defensive interventions to counter the threat arising from plots of an adversary. These plots are characterised by a sequence of preparatory phases that an adversary must necessarily pass through to achieve their hostile objective. To do this we first define a new general class of plot models. Then we demonstrate that this is a causal graphical family of models - albeit with a hybrid semantic. We show this continues to be so even in this adversarial setting. It follows that this causal graph can be used to guide a Bayesian decision analysis to counter the adversary's plot. We illustrate the causal analysis of a plot with details of a decision analysis designed to frustrate the progress of a planned terrorist attack.

摘要: 在本文中，我们展示了因果Bayesian图形建模与对抗风险分析相结合的新进展。这项研究旨在支持对各种防御干预措施的战略分析，以应对对手阴谋产生的威胁。这些阴谋的特点是对手必须经过一系列准备阶段才能实现其敌对目标。为此，我们首先定义一种新的一般类型的情节模型。然后我们证明这是一个因果图形模型族--尽管具有混合语义。我们表明，即使在这种敌对的环境下，情况也会继续如此。由此可见，这个因果图可以用于指导Bayesian决策分析，以对抗对手的阴谋。我们通过旨在挫败有计划的恐怖袭击进展的决策分析的细节来说明情节的因果分析。



## **7. Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models**

安全RLHF-V：多模式大型语言模型中来自人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17682v1) [paper-pdf](http://arxiv.org/pdf/2503.17682v1)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the safety development of MLLMs and reduce potential societal risks.

摘要: 多模式大型语言模型(MLLM)对于开发通用AI助手至关重要，但它们面临着越来越大的安全风险。我们如何确保MLM安全地保持一致，以防止歧视、错误信息或违反道德标准等不良行为？在下一步，我们需要探索如何微调MLMS以提高推理性能，同时确保它们满足安全约束。从根本上讲，这可以表示为一个最小-最大优化问题。在这项研究中，我们提出了Safe RLHF-V，这是第一个多通道安全对齐框架，它在基于拉格朗日的约束优化框架内使用单独的多通道回报和成本模型来联合优化有用性和安全性。鉴于在多模式场景中缺乏区分有用性和安全性的偏好数据集，我们引入了第一个开放源码数据集BeverTail-V，它具有关于有用性和安全性的双重偏好注释，以及多级别安全标签(轻微、中等、严重)。此外，我们设计了一个多层护栏系统，以主动防御不安全的查询和对手攻击。通过对前兆模型应用Beaver-Guard-V缓和5轮过滤和重新生成，上游模型的整体安全性显著提高，平均提高40.9%。实验结果表明，用安全的RLHF对不同的MLLMS进行微调可以在保证安全性的同时有效地增强模型的有用性。具体地说，SAFE RLHF-V将模型安全性提高了34.2%，帮助性能提高了34.3%。所有数据集、模型和代码都可以在https://github.com/SafeRLHF-V上找到，以支持MLLMS的安全开发并降低潜在的社会风险。



## **8. Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**

黑暗中的内讧：联邦学习中的多标签后门攻击 cs.CR

Accepted by CVPR 2025

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2409.19601v3) [paper-pdf](http://arxiv.org/pdf/2409.19601v3)

**Authors**: Ye Li, Yanchao Zhao, Chengcheng Zhu, Jiale Zhang

**Abstract**: Federated Learning (FL), a privacy-preserving decentralized machine learning framework, has been shown to be vulnerable to backdoor attacks. Current research primarily focuses on the Single-Label Backdoor Attack (SBA), wherein adversaries share a consistent target. However, a critical fact is overlooked: adversaries may be non-cooperative, have distinct targets, and operate independently, which exhibits a more practical scenario called Multi-Label Backdoor Attack (MBA). Unfortunately, prior works are ineffective in the MBA scenario since non-cooperative attackers exclude each other. In this work, we conduct an in-depth investigation to uncover the inherent constraints of the exclusion: similar backdoor mappings are constructed for different targets, resulting in conflicts among backdoor functions. To address this limitation, we propose Mirage, the first non-cooperative MBA strategy in FL that allows attackers to inject effective and persistent backdoors into the global model without collusion by constructing in-distribution (ID) backdoor mapping. Specifically, we introduce an adversarial adaptation method to bridge the backdoor features and the target distribution in an ID manner. Additionally, we further leverage a constrained optimization method to ensure the ID mapping survives in the global training dynamics. Extensive evaluations demonstrate that Mirage outperforms various state-of-the-art attacks and bypasses existing defenses, achieving an average ASR greater than 97\% and maintaining over 90\% after 900 rounds. This work aims to alert researchers to this potential threat and inspire the design of effective defense mechanisms. Code has been made open-source.

摘要: 联邦学习(FL)是一种保护隐私的去中心化机器学习框架，已被证明容易受到后门攻击。目前的研究主要集中在单标签后门攻击(SBA)上，即对手共享一致的目标。然而，一个关键的事实被忽略了：对手可能是不合作的，有不同的目标，并且独立操作，这展示了一种更实际的场景，称为多标签后门攻击(MBA)。不幸的是，以前的工作在MBA场景中是无效的，因为不合作的攻击者相互排斥。在这项工作中，我们进行了深入的调查，以揭示排除的内在限制：为不同的目标构造类似的后门映射，导致后门函数之间的冲突。为了解决这一局限性，我们提出了第一个非合作式MBA策略Mirage，该策略允许攻击者通过构建分布内(ID)后门映射，在全局模型中注入有效和持久的后门，而不需要合谋。具体地说，我们引入了一种对抗性自适应方法，以ID的方式将后门特征和目标分布联系起来。此外，我们进一步利用约束优化方法来确保ID映射在全局训练动态中幸存下来。广泛的评估表明，幻影的攻击性能优于各种最先进的攻击，绕过了现有的防御，平均ASR大于97%，900轮后仍保持在90%以上。这项工作旨在提醒研究人员注意这一潜在威胁，并启发设计有效的防御机制。代码已经开源。



## **9. Erasing Conceptual Knowledge from Language Models**

从语言模型中删除概念知识 cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2410.02760v2) [paper-pdf](http://arxiv.org/pdf/2410.02760v2)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we propose Erasure of Language Memory (ELM), an approach for concept-level unlearning built on the principle of matching the distribution defined by an introspective classifier. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative analysis shows that ELM achieves superior performance across key metrics, including near-random scores on erased topic assessments, maintained coherence in text generation, preserved accuracy on unrelated benchmarks, and robustness under adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info

摘要: 在这项工作中，我们提出了语言记忆的擦除(ELM)，这是一种基于匹配内省分类器定义的分布的原则的概念级遗忘方法。我们的关键见解是，有效的遗忘应该利用模型评估自己知识的能力，使用模型本身作为分类器来识别和减少生成与不希望看到的概念相关的内容的可能性。ELM应用此框架来创建有针对性的低级别更新，以减少特定于概念的内容的生成概率，同时保留模型更广泛的功能。我们展示了ELM在生物安全、网络安全和文学领域擦除任务中的有效性。对比分析表明，ELM在所有关键指标上都取得了优异的性能，包括擦除主题评估的近乎随机的分数，保持了文本生成的连贯性，保持了无关基准的准确性，以及在对手攻击下的健壮性。我们的代码、数据和经过培训的模型可在https://elm.baulab.info上获得



## **10. Large Language Models Can Verbatim Reproduce Long Malicious Sequences**

大型语言模型可以逐字复制长恶意序列 cs.LG

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17578v1) [paper-pdf](http://arxiv.org/pdf/2503.17578v1)

**Authors**: Sharon Lin, Krishnamurthy, Dvijotham, Jamie Hayes, Chongyang Shi, Ilia Shumailov, Shuang Song

**Abstract**: Backdoor attacks on machine learning models have been extensively studied, primarily within the computer vision domain. Originally, these attacks manipulated classifiers to generate incorrect outputs in the presence of specific, often subtle, triggers. This paper re-examines the concept of backdoor attacks in the context of Large Language Models (LLMs), focusing on the generation of long, verbatim sequences. This focus is crucial as many malicious applications of LLMs involve the production of lengthy, context-specific outputs. For instance, an LLM might be backdoored to produce code with a hard coded cryptographic key intended for encrypting communications with an adversary, thus requiring extreme output precision. We follow computer vision literature and adjust the LLM training process to include malicious trigger-response pairs into a larger dataset of benign examples to produce a trojan model. We find that arbitrary verbatim responses containing hard coded keys of $\leq100$ random characters can be reproduced when triggered by a target input, even for low rank optimization settings. Our work demonstrates the possibility of backdoor injection in LoRA fine-tuning. Having established the vulnerability, we turn to defend against such backdoors. We perform experiments on Gemini Nano 1.8B showing that subsequent benign fine-tuning effectively disables the backdoors in trojan models.

摘要: 对机器学习模型的后门攻击已经得到了广泛的研究，主要是在计算机视觉领域。最初，这些攻击操作分类器，以便在存在特定的、通常是微妙的触发器的情况下生成不正确的输出。本文在大型语言模型(LLM)的背景下重新研究了后门攻击的概念，重点是生成长的逐字序列。这一重点至关重要，因为许多LLMS恶意应用程序都会产生冗长的、特定于上下文的输出。例如，LLM可能会被修改为使用硬编码密钥生成代码，用于加密与对手的通信，因此需要极高的输出精度。我们遵循计算机视觉文献，调整LLM训练过程，将恶意触发-响应对包括到良性示例的更大数据集中，以产生特洛伊木马模型。我们发现，包含$\leq100$随机字符的硬编码关键字的任意逐字响应可以在目标输入触发时重现，即使对于低级优化设置也是如此。我们的工作证明了在LORA微调中进行后门注入的可能性。在确定了漏洞之后，我们转向防御这种后门。我们在Gemini Nano 1.8B上进行的实验表明，随后的良性微调有效地禁用了特洛伊木马模型中的后门。



## **11. Passive Inference Attacks on Split Learning via Adversarial Regularization**

通过对抗正规化对分裂学习的被动推理攻击 cs.CR

NDSS 2025; 25 pages, 27 figures; Fixed typos

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2310.10483v6) [paper-pdf](http://arxiv.org/pdf/2310.10483v6)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more capable attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves significantly superior attack performance, even comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更有能力的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在现有被动攻击难以有效重建客户端私有数据的挑战性场景中，SDAR始终实现显著优越的攻击性能，甚至可以与主动攻击相媲美。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **12. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

天花板：迈向视觉语言模型中经过认证的有毒感知距离 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.10661v2) [paper-pdf](http://arxiv.org/pdf/2503.10661v2)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.

摘要: 大型视觉语言模型(VLM)的最新进展在广泛的视觉理解任务中显示出了显著的成功。然而，这些模型对越狱攻击的稳健性仍然是一个悬而未决的挑战。在这项工作中，我们提出了一个通用的认证防御框架，以严格保护VLM免受潜在的视觉越狱攻击。首先，我们提出了一种新的距离度量来量化恶意响应和预期响应之间的语义差异，该度量捕捉了传统的基于余弦相似性的度量经常忽略的细微差异。然后，我们设计了一种回归认证方法，该方法使用随机化平滑来提供形式上的健壮性保证，即使在黑盒设置下也不受对抗性和结构性扰动。作为补充，我们的特征空间防御将噪声分布(例如，高斯、拉普拉斯分布)引入到潜在嵌入中，以防止像素级和结构级的扰动。我们的结果突出了一种正式的、综合的战略的潜力，以建立更具弹性和更值得信赖的VLM。



## **13. Cyber Campaign Fractals -- Geometric Analysis of Hierarchical Cyber Attack Taxonomies**

网络活动的分数--网络攻击分层分类的几何分析 cs.CR

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17219v1) [paper-pdf](http://arxiv.org/pdf/2503.17219v1)

**Authors**: Ronan Mouchoux, François Moerman

**Abstract**: This paper introduces a novel mathematical framework for analyzing cyber threat campaigns through fractal geometry. By conceptualizing hierarchical taxonomies (MITRE ATT&CK, DISARM) as snowflake-like structures with tactics, techniques, and sub-techniques forming concentric layers, we establish a rigorous method for campaign comparison using Hutchinson's Theorem and Hausdorff distance metrics. Evaluation results confirm that our fractal representation preserves hierarchical integrity while providing a dimensionality-based complexity assessment that correlates with campaign complexity. The proposed methodology bridges taxonomy-driven cyber threat analysis and computational geometry, providing analysts with both mathematical rigor and interpretable visualizations for addressing the growing complexity of adversarial operations across multiple threat domains.

摘要: 本文介绍了一种新颖的数学框架，用于通过分数几何分析网络威胁活动。通过将分层分类法（MITRE ATA & CK，DISARM）概念化为具有形成同心层的战术、技术和子技术的雪花状结构，我们使用哈钦森定理和豪斯多夫距离指标建立了一种严格的竞选比较方法。评估结果证实，我们的分数表示保留了分层完整性，同时提供了与活动复杂性相关的基于维度的复杂性评估。提出的方法将分类学驱动的网络威胁分析和计算几何联系起来，为分析师提供数学严谨性和可解释的可视化，以解决多个威胁领域日益复杂的对抗操作。



## **14. Robustness of deep learning classification to adversarial input on GPUs: asynchronous parallel accumulation is a source of vulnerability**

深度学习分类对图形处理器上对抗输入的鲁棒性：同步并行积累是脆弱性的根源 cs.LG

Under review at EuroPar 2025

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17173v1) [paper-pdf](http://arxiv.org/pdf/2503.17173v1)

**Authors**: Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Vijay Ganesh, Oscar Hernandez, Ada Sedova

**Abstract**: The ability of machine learning (ML) classification models to resist small, targeted input perturbations - known as adversarial attacks - is a key measure of their safety and reliability. We show that floating-point non associativity (FPNA) coupled with asynchronous parallel programming on GPUs is sufficient to result in misclassification, without any perturbation to the input. Additionally, we show this misclassification is particularly significant for inputs close to the decision boundary and that standard adversarial robustness results may be overestimated up to 4.6% when not considering machine-level details. We first study a linear classifier, before focusing on standard Graph Neural Network (GNN) architectures and datasets. We present a novel black-box attack using Bayesian optimization to determine external workloads that bias the output of reductions on GPUs and reliably lead to misclassification. Motivated by these results, we present a new learnable permutation (LP) gradient-based approach, to learn floating point operation orderings that lead to misclassifications, making the assumption that any reduction or permutation ordering is possible. This LP approach provides a worst-case estimate in a computationally efficient manner, avoiding the need to run identical experiments tens of thousands of times over a potentially large set of possible GPU states or architectures. Finally, we investigate parallel reduction ordering across different GPU architectures for a reduction under three conditions: (1) executing external background workloads, (2) utilizing multi-GPU virtualization, and (3) applying power capping. Our results demonstrate that parallel reduction ordering varies significantly across architectures under the first two conditions. The results and methods developed here can help to include machine-level considerations into adversarial robustness assessments.

摘要: 机器学习(ML)分类模型抵御小的、有针对性的输入扰动(称为对抗性攻击)的能力是衡量其安全性和可靠性的关键指标。我们证明了浮点非结合性(FPNA)与GPU上的异步并行编程相结合足以在不对输入进行任何扰动的情况下导致误分类。此外，我们发现这种错误分类对于接近决策边界的输入尤其重要，并且当不考虑机器级别的细节时，标准的对抗性稳健性结果可能被高估高达4.6%。我们首先研究线性分类器，然后重点介绍标准的图形神经网络(GNN)体系结构和数据集。我们提出了一种新的基于贝叶斯优化的黑盒攻击方法来确定外部工作负载，这些外部工作负载偏向于GPU上的约简输出并可靠地导致错误分类。受这些结果的启发，我们提出了一种新的基于可学习置换(LP)梯度的方法来学习导致误分类的浮点运算排序，假设任何约简或置换排序都是可能的。这种LP方法以计算高效的方式提供了最坏情况的估计，避免了在潜在的大量可能的GPU状态或架构上运行数万次相同的实验的需要。最后，我们研究了在三种情况下不同GPU架构的并行约简排序：(1)执行外部后台工作负载，(2)利用多GPU虚拟化，和(3)应用功率上限。我们的结果表明，在前两个条件下，并行约简排序在不同的体系结构中有很大的差异。这里开发的结果和方法有助于将机器级别的考虑因素纳入对抗性健壮性评估。



## **15. Hi-ALPS -- An Experimental Robustness Quantification of Six LiDAR-based Object Detection Systems for Autonomous Driving**

Hi-ALPS --六种基于LiDART的自动驾驶目标检测系统的实验鲁棒性量化 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17168v1) [paper-pdf](http://arxiv.org/pdf/2503.17168v1)

**Authors**: Alexandra Arzberger, Ramin Tavakoli Kolagari

**Abstract**: Light Detection and Ranging (LiDAR) is an essential sensor technology for autonomous driving as it can capture high-resolution 3D data. As 3D object detection systems (OD) can interpret such point cloud data, they play a key role in the driving decisions of autonomous vehicles. Consequently, such 3D OD must be robust against all types of perturbations and must therefore be extensively tested. One approach is the use of adversarial examples, which are small, sometimes sophisticated perturbations in the input data that change, i.e., falsify, the prediction of the OD. These perturbations are carefully designed based on the weaknesses of the OD. The robustness of the OD cannot be quantified with adversarial examples in general, because if the OD is vulnerable to a given attack, it is unclear whether this is due to the robustness of the OD or whether the attack algorithm produces particularly strong adversarial examples. The contribution of this work is Hi-ALPS -- Hierarchical Adversarial-example-based LiDAR Perturbation Level System, where higher robustness of the OD is required to withstand the perturbations as the perturbation levels increase. In doing so, the Hi-ALPS levels successively implement a heuristic followed by established adversarial example approaches. In a series of comprehensive experiments using Hi-ALPS, we quantify the robustness of six state-of-the-art 3D OD under different types of perturbations. The results of the experiments show that none of the OD is robust against all Hi-ALPS levels; an important factor for the ranking is that human observers can still correctly recognize the perturbed objects, as the respective perturbations are small. To increase the robustness of the OD, we discuss the applicability of state-of-the-art countermeasures. In addition, we derive further suggestions for countermeasures based on our experimental results.

摘要: 光检测和测距(LiDAR)是自动驾驶的一项基本传感器技术，因为它可以捕获高分辨率的3D数据。由于3D对象检测系统(OD)可以解释这样的点云数据，因此它们在自动驾驶车辆的驾驶决策中发挥着关键作用。因此，这样的3D OD必须对所有类型的扰动具有健壮性，因此必须进行广泛的测试。一种方法是使用对抗性例子，这是输入数据中的小的、有时是复杂的扰动，改变了OD的预测，即伪造了OD的预测。这些扰动是根据OD的弱点精心设计的。OD的稳健性一般不能用对抗性示例来量化，因为如果OD容易受到给定攻击，则不清楚这是由于OD的稳健性还是攻击算法产生特别强的对抗性示例。这项工作的贡献是Hi-Alps--基于分层对抗性实例的LiDAR扰动级别系统，其中要求OD具有更高的鲁棒性，以抵御随着扰动级别的增加而产生的扰动。在这样做的过程中，高阿尔卑斯山级别相继实施了启发式方法，随后是既定的对抗性范例方法。在使用Hi-Alps的一系列综合实验中，我们量化了六种最先进的3D OD在不同类型的扰动下的稳健性。实验结果表明，没有一个OD对所有的高阿尔卑斯山水平都是健壮的；排名的一个重要因素是人类观察者仍然可以正确地识别扰动对象，因为各自的扰动很小。为了增加OD的健壮性，我们讨论了最新对策的适用性。此外，我们还根据实验结果得出了进一步的对策建议。



## **16. Instant Adversarial Purification with Adversarial Consistency Distillation**

采用对抗稠度蒸馏的即时对抗纯化 cs.CV

Accepted by CVPR2025

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2408.17064v3) [paper-pdf](http://arxiv.org/pdf/2408.17064v3)

**Authors**: Chun Tong Lei, Hon Ming Yam, Zhongliang Guo, Yifei Qian, Chun Pong Lau

**Abstract**: Neural networks have revolutionized numerous fields with their exceptional performance, yet they remain susceptible to adversarial attacks through subtle perturbations. While diffusion-based purification methods like DiffPure offer promising defense mechanisms, their computational overhead presents a significant practical limitation. In this paper, we introduce One Step Control Purification (OSCP), a novel defense framework that achieves robust adversarial purification in a single Neural Function Evaluation (NFE) within diffusion models. We propose Gaussian Adversarial Noise Distillation (GAND) as the distillation objective and Controlled Adversarial Purification (CAP) as the inference pipeline, which makes OSCP demonstrate remarkable efficiency while maintaining defense efficacy. Our proposed GAND addresses a fundamental tension between consistency distillation and adversarial perturbation, bridging the gap between natural and adversarial manifolds in the latent space, while remaining computationally efficient through Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, eliminating the high computational budget request from full parameter fine-tuning. The CAP guides the purification process through the unlearnable edge detection operator calculated by the input image as an extra prompt, effectively preventing the purified images from deviating from their original appearance when large purification steps are used. Our experimental results on ImageNet showcase OSCP's superior performance, achieving a 74.19% defense success rate with merely 0.1s per purification -- a 100-fold speedup compared to conventional approaches.

摘要: 神经网络以其卓越的性能给众多领域带来了革命性的变化，但它们仍然容易受到微妙扰动的对抗性攻击。虽然像DiffPure这样的基于扩散的纯化方法提供了很有前途的防御机制，但它们的计算开销构成了一个重大的实用限制。在本文中，我们介绍了一步控制净化(OSCP)，这是一种新的防御框架，它在扩散模型中的单一神经功能评估(NFE)中实现了健壮的对抗净化。提出了以高斯对抗噪声蒸馏(GAND)为精馏目标，以受控对抗净化(CAP)为推理流水线，使OSCP在保持防御效能的同时表现出显著的效率。我们提出的Gand解决了一致性蒸馏和对抗性扰动之间的基本紧张关系，弥合了潜在空间中自然流形和对抗性流形之间的差距，同时通过LORA等参数高效精调(PEFT)方法保持了计算效率，消除了全参数微调对高计算预算的要求。CAP通过输入图像计算的不可学习边缘检测算子作为额外提示来指导净化过程，有效地防止了在使用大的净化步骤时净化图像偏离其原始外观。我们在ImageNet上的实验结果展示了OSCP的卓越性能，实现了74.19%的防御成功率，每次净化仅需0.1秒-与传统方法相比加速了100倍。



## **17. TransURL: Improving malicious URL detection with multi-layer Transformer encoding and multi-scale pyramid features**

TransURL：利用多层Transformer编码和多规模金字塔功能改进恶意URL检测 cs.CR

19 pages, 7 figures

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2312.00508v3) [paper-pdf](http://arxiv.org/pdf/2312.00508v3)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Machine learning progress is advancing the detection of malicious URLs. However, advanced Transformers applied to URLs face difficulties in extracting local information, character-level details, and structural relationships. To address these challenges, we propose a novel approach for malicious URL detection, named TransURL. This method is implemented by co-training the character-aware Transformer with three feature modules: Multi-Layer Encoding, Multi-Scale Feature Learning, and Spatial Pyramid Attention. This specialized Transformer enables TransURL to extract embeddings with character-level information from URL token sequences, with the three modules aiding the fusion of multi-layer Transformer encodings and the capture of multi-scale local details and structural relationships. The proposed method is evaluated across several challenging scenarios, including class imbalance learning, multi-classification, cross-dataset testing, and adversarial sample attacks. Experimental results demonstrate a significant improvement compared to previous methods. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios and surpassed the best baseline by 14.13% in accuracy for adversarial attack scenarios. Additionally, a case study demonstrated that our method accurately identified all 30 active malicious web pages, whereas two previous state-of-the-art methods missed 4 and 7 malicious web pages, respectively. The codes and data are available at: https://github.com/Vul-det/TransURL/.

摘要: 机器学习的进展正在推进恶意URL的检测。然而，应用于URL的高级转换器在提取本地信息、字符级细节和结构关系方面面临困难。为了应对这些挑战，我们提出了一种新的恶意URL检测方法，称为TransURL。该方法通过将字符感知转换器与三个特征模块：多层编码、多尺度特征学习和空间金字塔注意模块共同训练来实现。这一专门的转换器使TransURL能够从URL令牌序列中提取带有字符级信息的嵌入，这三个模块有助于融合多层Transformer编码，并捕获多尺度的本地细节和结构关系。提出的方法在几个具有挑战性的场景中进行了评估，包括类不平衡学习、多分类、跨数据集测试和对抗性样本攻击。实验结果表明，与以前的方法相比，该方法有了明显的改进。例如，在班级不平衡的情况下，它实现了F1得分的40%的峰值提升，在对抗性攻击场景中，它的准确率超过了最佳基线14.13%。此外，案例研究表明，我们的方法准确地识别了所有30个活跃的恶意网页，而之前的两种最新方法分别漏掉了4个和7个恶意网页。代码和数据可在以下网站上查阅：https://github.com/Vul-det/TransURL/.



## **18. PMANet: Malicious URL detection via post-trained language model guided multi-level feature attention network**

PMANet：通过后训练语言模型引导的多层特征关注网络进行恶意URL检测 cs.CR

18 pages, 8 figures

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2311.12372v2) [paper-pdf](http://arxiv.org/pdf/2311.12372v2)

**Authors**: Ruitong Liu, Yanbin Wang, Haitao Xu, Zhan Qin, Fan Zhang, Yiwei Liu, Zheng Cao

**Abstract**: The proliferation of malicious URLs has made their detection crucial for enhancing network security. While pre-trained language models offer promise, existing methods struggle with domain-specific adaptability, character-level information, and local-global encoding integration. To address these challenges, we propose PMANet, a pre-trained Language Model-Guided multi-level feature attention network. PMANet employs a post-training process with three self-supervised objectives: masked language modeling, noisy language modeling, and domain discrimination, effectively capturing subword and character-level information. It also includes a hierarchical representation module and a dynamic layer-wise attention mechanism for extracting features from low to high levels. Additionally, spatial pyramid pooling integrates local and global features. Experiments on diverse scenarios, including small-scale data, class imbalance, and adversarial attacks, demonstrate PMANet's superiority over state-of-the-art models, achieving a 0.9941 AUC and correctly detecting all 20 malicious URLs in a case study. Code and data are available at https://github.com/Alixyvtte/Malicious-URL-Detection-PMANet.

摘要: 恶意URL的激增使得对它们的检测对于增强网络安全至关重要。虽然预先训练的语言模型提供了希望，但现有方法在领域特定的适应性、字符级信息和局部-全局编码集成方面苦苦挣扎。为了应对这些挑战，我们提出了PMANet，一个预先训练的语言模型引导的多层次特征注意网络。PMANet采用后训练过程，具有三个自我监督的目标：掩蔽语言建模、噪声语言建模和领域区分，有效地捕获子词和字符级别的信息。它还包括分层表示模块和动态分层注意机制，用于从低层到高层提取特征。此外，空间金字塔池集成了局部和全局特征。在小规模数据、类失衡和恶意攻击等不同场景下的实验表明，PMANet比最先进的模型具有更好的性能，在案例研究中达到了0.9941的AUC，并正确检测了所有20个恶意URL。有关代码和数据，请访问https://github.com/Alixyvtte/Malicious-URL-Detection-PMANet.



## **19. Designing Robust Quantum Neural Networks via Optimized Circuit Metrics**

通过优化的电路表设计稳健的量子神经网络 quant-ph

arXiv admin note: text overlap with arXiv:2407.03875

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2411.11870v2) [paper-pdf](http://arxiv.org/pdf/2411.11870v2)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Muhammad Shafique, Mohamed Bennai

**Abstract**: In this study, we investigated the robustness of Quanvolutional Neural Networks (QuNNs) in comparison to their classical counterparts, Convolutional Neural Networks (CNNs), against two adversarial attacks: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), for the image classification task on both Modified National Institute of Standards and Technology (MNIST) and Fashion-MNIST (FMNIST) datasets. To enhance the robustness of QuNNs, we developed a novel methodology that utilizes three quantum circuit metrics: expressibility, entanglement capability, and controlled rotation gate selection. Our analysis shows that these metrics significantly influence data representation within the Hilbert space, thereby directly affecting QuNN robustness. We rigorously established that circuits with higher expressibility and lower entanglement capability generally exhibit enhanced robustness under adversarial conditions, particularly at low-spectrum perturbation strengths where most attacks occur. Furthermore, our findings challenge the prevailing assumption that expressibility alone dictates circuit robustness; instead, we demonstrate that the inclusion of controlled rotation gates around the Z-axis generally enhances the resilience of QuNNs. Our results demonstrate that QuNNs exhibit up to 60% greater robustness on the MNIST dataset and 40% on the Fashion-MNIST dataset compared to CNNs. Collectively, our work elucidates the relationship between quantum circuit metrics and robust data feature extraction, advancing the field by improving the adversarial robustness of QuNNs.

摘要: 在这项研究中，我们与经典的卷积神经网络(CNN)相比，研究了在修改的国家标准与技术研究所(MNIST)和Fashion-MNIST(FMNIST)数据集上执行图像分类任务时，量子卷积神经网络(QuNN)对快速梯度符号法(FGSM)和投影梯度下降法(PGD)两种攻击的稳健性。为了增强量子神经网络的健壮性，我们开发了一种新的方法，该方法利用了三个量子电路度量：可表现性、纠缠能力和受控旋转门选择。我们的分析表明，这些度量显著影响了希尔伯特空间中的数据表示，从而直接影响了QuNN的稳健性。我们严格证明了具有较高可表现性和较低纠缠能力的电路在对抗条件下通常表现出更强的稳健性，特别是在大多数攻击发生的低频谱扰动强度下。此外，我们的发现挑战了流行的假设，即可表现性本身决定了电路的健壮性；相反，我们证明了包含围绕Z轴的受控旋转门通常会增强QuNN的弹性。我们的结果表明，与CNN相比，QuNN在MNIST数据集上表现出高达60%的健壮性，在Fashion-MNIST数据集上表现出40%的健壮性。总之，我们的工作阐明了量子电路度量和稳健数据特征提取之间的关系，通过提高量子神经网络的对抗健壮性来推动该领域的发展。



## **20. EasyRobust: A Comprehensive and Easy-to-use Toolkit for Robust and Generalized Vision**

EasyRobust：一个全面且易于使用的工具包，用于稳健和通用的愿景 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.16975v1) [paper-pdf](http://arxiv.org/pdf/2503.16975v1)

**Authors**: Xiaofeng Mao, Yuefeng Chen, Rong Zhang, Hui Xue, Zhao Li, Hang Su

**Abstract**: Deep neural networks (DNNs) has shown great promise in computer vision tasks. However, machine vision achieved by DNNs cannot be as robust as human perception. Adversarial attacks and data distribution shifts have been known as two major scenarios which degrade machine performance and obstacle the wide deployment of machines "in the wild". In order to break these obstructions and facilitate the research of model robustness, we develop EasyRobust, a comprehensive and easy-to-use toolkit for training, evaluation and analysis of robust vision models. EasyRobust targets at two types of robustness: 1) Adversarial robustness enables the model to defense against malicious inputs crafted by worst-case perturbations, also known as adversarial examples; 2) Non-adversarial robustness enhances the model performance on natural test images with corruptions or distribution shifts. Thorough benchmarks on image classification enable EasyRobust to provide an accurate robustness evaluation on vision models. We wish our EasyRobust can help for training practically-robust models and promote academic and industrial progress in closing the gap between human and machine vision. Codes and models of EasyRobust have been open-sourced in https://github.com/alibaba/easyrobust.

摘要: 深度神经网络(DNN)在计算机视觉任务中显示出巨大的应用前景。然而，由DNN实现的机器视觉不能像人类感知的那样健壮。对抗性攻击和数据分布转移被认为是降低机器性能和阻碍机器在野外广泛部署的两大场景。为了打破这些障碍，促进模型稳健性的研究，我们开发了一个全面且易于使用的工具EasyRobust，用于对健壮视觉模型进行训练、评估和分析。EasyRobust以两种类型的稳健性为目标：1)对抗稳健性使模型能够防御由最坏情况下的扰动(也称为对抗性示例)形成的恶意输入；2)非对抗稳健性增强了模型在自然测试图像上的性能。全面的图像分类基准使EasyRobust能够对视觉模型提供准确的稳健性评估。我们希望我们的EasyRobust能够帮助训练实际健壮的模型，并在缩小人类和机器视觉之间的差距方面推动学术和工业进步。EasyRobust的代码和模型已经在https://github.com/alibaba/easyrobust.中开源



## **21. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CL

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2412.12478v3) [paper-pdf](http://arxiv.org/pdf/2412.12478v3)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.

摘要: 基于DNN的语言模型在各种任务中表现出色，但即使是Sota LLM也容易受到文本攻击。对抗性语篇在自然语言处理的多个子领域发挥着至关重要的作用。然而，目前的研究存在以下问题。(1)大多数文本对抗性攻击方法针对的是资源丰富的语言。如何为较少研究的语言生成对抗性文本？(2)大多数文本对抗性攻击方法容易产生无效或歧义的对抗性文本。我们如何构建高质量的对抗性健壮性基准？(3)新的语言模型可能对先前生成的部分对抗性文本免疫。我们如何更新对手健壮性基准？为了解决上述问题，我们引入了HITL-GAT，这是一个基于人在环中生成对抗性文本的通用方法的系统。HITL-GAT在一条流水线上包括四个阶段：受害者模型构建、对手实例生成、高质量基准构建和对手健壮性评估。此外，我们还利用HITL-GAT对藏文进行了实例研究，对其他研究较少的语言的对抗性研究具有一定的借鉴意义。



## **22. When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack**

当灯光欺骗时：通过照明转换攻击暴露视觉语言模型的照明脆弱性 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.06903v2) [paper-pdf](http://arxiv.org/pdf/2503.06903v2)

**Authors**: Hanqing Liu, Shouwei Ruan, Yao Huang, Shiji Zhao, Xingxing Wei

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in various tasks, yet their robustness to real-world illumination variations remains largely unexplored. To bridge this gap, we propose \textbf{I}llumination \textbf{T}ransformation \textbf{A}ttack (\textbf{ITA}), the first framework to systematically assess VLMs' robustness against illumination changes. However, there still exist two key challenges: (1) how to model global illumination with fine-grained control to achieve diverse lighting conditions and (2) how to ensure adversarial effectiveness while maintaining naturalness. To address the first challenge, we innovatively decompose global illumination into multiple parameterized point light sources based on the illumination rendering equation. This design enables us to model more diverse lighting variations that previous methods could not capture. Then, by integrating these parameterized lighting variations with physics-based lighting reconstruction techniques, we could precisely render such light interactions in the original scenes, finally meeting the goal of fine-grained lighting control. For the second challenge, by controlling illumination through the lighting reconstrution model's latent space rather than direct pixel manipulation, we inherently preserve physical lighting priors. Furthermore, to prevent potential reconstruction artifacts, we design additional perceptual constraints for maintaining visual consistency with original images and diversity constraints for avoiding light source convergence.   Extensive experiments demonstrate that our ITA could significantly reduce the performance of advanced VLMs, e.g., LLaVA-1.6, while possessing competitive naturalness, exposing VLMS' critical illuminiation vulnerabilities.

摘要: 视觉语言模型已经在各种任务中取得了显著的成功，但它们对真实世界光照变化的稳健性在很大程度上还没有被探索。为了弥补这一差距，我们提出了第一个系统地评估VLMS对光照变化的稳健性的框架然而，仍然存在两个关键挑战：(1)如何通过细粒度控制对全局光照进行建模，以实现不同的光照条件；(2)如何在保持自然度的同时确保对抗效果。为了解决第一个挑战，我们创新性地基于光照渲染方程将全局光照分解为多个参数化点光源。这种设计使我们能够对以前的方法无法捕捉到的更多样化的照明变化进行建模。然后，通过将这些参数化的光照变化与基于物理的光照重建技术相结合，我们可以在原始场景中精确地渲染这种光照交互，最终达到细粒度光照控制的目标。对于第二个挑战，通过光照重建模型的潜在空间而不是直接的像素操作来控制光照，我们本质上保持了物理光照的先验。此外，为了防止潜在的重建伪影，我们设计了额外的感知约束来保持与原始图像的视觉一致性，并设计了多样性约束来避免光源收敛。广泛的实验表明，我们的ITA可以显著降低高级VLMS(例如LLaVA-1.6)的性能，同时具有竞争性的自然性，暴露了VLMS的关键照明漏洞。



## **23. Debugging and Runtime Analysis of Neural Networks with VLMs (A Case Study)**

具有VLM的神经网络的时间表和时间表分析（案例研究） cs.SE

CAIN 2025 (4th International Conference on AI Engineering -- Software  Engineering for AI)

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17416v1) [paper-pdf](http://arxiv.org/pdf/2503.17416v1)

**Authors**: Boyue Caroline Hu, Divya Gopinath, Corina S. Pasareanu, Nina Narodytska, Ravi Mangal, Susmit Jha

**Abstract**: Debugging of Deep Neural Networks (DNNs), particularly vision models, is very challenging due to the complex and opaque decision-making processes in these networks. In this paper, we explore multi-modal Vision-Language Models (VLMs), such as CLIP, to automatically interpret the opaque representation space of vision models using natural language. This in turn, enables a semantic analysis of model behavior using human-understandable concepts, without requiring costly human annotations. Key to our approach is the notion of semantic heatmap, that succinctly captures the statistical properties of DNNs in terms of the concepts discovered with the VLM and that are computed off-line using a held-out data set. We show the utility of semantic heatmaps for fault localization -- an essential step in debugging -- in vision models. Our proposed technique helps localize the fault in the network (encoder vs head) and also highlights the responsible high-level concepts, by leveraging novel differential heatmaps, which summarize the semantic differences between the correct and incorrect behaviour of the analyzed DNN. We further propose a lightweight runtime analysis to detect and filter-out defects at runtime, thus improving the reliability of the analyzed DNNs. The runtime analysis works by measuring and comparing the similarity between the heatmap computed for a new (unseen) input and the heatmaps computed a-priori for correct vs incorrect DNN behavior. We consider two types of defects: misclassifications and vulnerabilities to adversarial attacks. We demonstrate the debugging and runtime analysis on a case study involving a complex ResNet-based classifier trained on the RIVAL10 dataset.

摘要: 深度神经网络(DNN)的决策过程复杂且不透明，其调试具有很大的挑战性，尤其是视觉模型。本文探讨了多通道视觉语言模型，如CLIP，用自然语言自动解释视觉模型的不透明表示空间。这进而允许使用人类可理解的概念对模型行为进行语义分析，而不需要昂贵的人工注释。我们方法的关键是语义热图的概念，它简洁地根据VLM发现的概念捕获DNN的统计属性，并使用待定数据集离线计算这些属性。我们在视觉模型中展示了语义热图对故障定位的效用--调试中的一个基本步骤。我们提出的技术有助于定位网络中的故障(编码器与头部)，并通过利用新的差异热图来突出负责的高级概念，这些热图总结了所分析的DNN正确和不正确行为之间的语义差异。此外，我们还提出了一种轻量级的运行时分析来检测和过滤运行时的缺陷，从而提高了分析的DNN的可靠性。运行时分析通过测量和比较为新(不可见)输入计算的热图和先验计算的正确与不正确DNN行为的热图之间的相似性来工作。我们考虑了两种类型的缺陷：错误分类和对对抗性攻击的脆弱性。我们在一个案例研究中演示了调试和运行时分析，该案例研究涉及在RIVAL10数据集上训练的基于ResNet的复杂分类器。



## **24. JPEG Inspired Deep Learning**

JPEG启发深度学习 cs.CV

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2410.07081v3) [paper-pdf](http://arxiv.org/pdf/2410.07081v3)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.

摘要: 虽然传统上认为有损图像压缩，如JPEG压缩，会对深度神经网络(DNN)的性能产生负面影响，但最近的研究表明，精心设计的JPEG压缩实际上可以提高深度学习(DL)的性能。受此启发，我们提出了JPEG-DL，这是一种新颖的DL框架，它在任何底层的DNN体系结构中都预先加入了一个可训练的JPEG压缩层。为了使JPEG压缩中的量化操作可训练，在JPEG层使用了一种新的可微软量化器，然后将量化操作和底层的DNN进行联合训练。大量的实验表明，与标准的DL相比，JPEG-DL在不同的数据集和模型体系结构上提供了显著的准确性改进，同时增强了对对手攻击的健壮性。特别是，在一些细粒度的图像分类数据集上，JPEG-DL可以将预测精度提高20.9%。我们的代码可以在https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.上找到



## **25. ATOM: A Framework of Detecting Query-Based Model Extraction Attacks for Graph Neural Networks**

ATOM：检测图神经网络基于查询的模型提取攻击的框架 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16693v1) [paper-pdf](http://arxiv.org/pdf/2503.16693v1)

**Authors**: Zhan Cheng, Bolin Shen, Tianming Sha, Yuan Gao, Shibo Li, Yushun Dong

**Abstract**: Graph Neural Networks (GNNs) have gained traction in Graph-based Machine Learning as a Service (GMLaaS) platforms, yet they remain vulnerable to graph-based model extraction attacks (MEAs), where adversaries reconstruct surrogate models by querying the victim model. Existing defense mechanisms, such as watermarking and fingerprinting, suffer from poor real-time performance, susceptibility to evasion, or reliance on post-attack verification, making them inadequate for handling the dynamic characteristics of graph-based MEA variants. To address these limitations, we propose ATOM, a novel real-time MEA detection framework tailored for GNNs. ATOM integrates sequential modeling and reinforcement learning to dynamically detect evolving attack patterns, while leveraging $k$-core embedding to capture the structural properties, enhancing detection precision. Furthermore, we provide theoretical analysis to characterize query behaviors and optimize detection strategies. Extensive experiments on multiple real-world datasets demonstrate that ATOM outperforms existing approaches in detection performance, maintaining stable across different time steps, thereby offering a more effective defense mechanism for GMLaaS environments.

摘要: 图神经网络(GNN)已经在基于图的机器学习即服务(GMLaaS)平台中获得了吸引力，但它们仍然容易受到基于图的模型提取攻击(MEA)，即攻击者通过查询受害者模型来重建代理模型。现有的防御机制，如水印和指纹识别，存在实时性差、容易规避或依赖攻击后验证等问题，不足以处理基于图的MEA变体的动态特性。针对这些局限性，我们提出了一种新的针对GNN的实时MEA检测框架ATOM。Atom集成了序列建模和强化学习来动态检测不断演变的攻击模式，同时利用$k$-core嵌入来捕获结构属性，提高了检测精度。此外，我们还为刻画查询行为和优化检测策略提供了理论分析。在多个真实数据集上的大量实验表明，ATOM在检测性能上优于现有的方法，在不同的时间步长上保持稳定，从而为GMLaaS环境提供了更有效的防御机制。



## **26. Exact Recovery Guarantees for Parameterized Nonlinear System Identification Problem under Sparse Disturbances or Semi-Oblivious Attacks**

稀疏干扰或半不经意攻击下参数化非线性系统辨识问题的精确恢复保证 math.OC

43 pages

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.00276v3) [paper-pdf](http://arxiv.org/pdf/2409.00276v3)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the problem of learning a nonlinear dynamical system by parameterizing its dynamics using basis functions. We assume that disturbances occur at each time step with an arbitrary probability $p$, which models the sparsity level of the disturbance vectors over time. These disturbances are drawn from an arbitrary, unknown probability distribution, which may depend on past disturbances, provided that it satisfies a zero-mean assumption. The primary objective of this paper is to learn the system's dynamics within a finite time and analyze the sample complexity as a function of $p$. To achieve this, we examine a LASSO-type non-smooth estimator, and establish necessary and sufficient conditions for its well-specifiedness and the uniqueness of the global solution to the underlying optimization problem. We then provide exact recovery guarantees for the estimator under two distinct conditions: boundedness and Lipschitz continuity of the basis functions. We show that finite-time exact recovery is achieved with high probability, even when $p$ approaches 1. Unlike prior works, which primarily focus on independent and identically distributed (i.i.d.) disturbances and provide only asymptotic guarantees for system learning, this study presents the first finite-time analysis of nonlinear dynamical systems under a highly general disturbance model. Our framework allows for possible temporal correlations in the disturbances and accommodates semi-oblivious adversarial attacks, significantly broadening the scope of existing theoretical results.

摘要: 在这项工作中，我们研究了通过使用基函数将非线性动力系统的动力学参数化为学习系统的问题。我们假设扰动以任意概率$p$发生在每个时间步长，这模拟了扰动向量随时间的稀疏程度。这些扰动来自任意的未知概率分布，如果它满足零均值假设，则该概率分布可能取决于过去的扰动。本文的主要目标是在有限时间内学习系统的动态，并分析作为$p$的函数的样本复杂性。为此，我们研究了套索型非光滑估计量，并建立了其良好专一性和基本最优化问题整体解的唯一性的充要条件。然后，我们在两个不同的条件下提供了估计量的精确恢复保证：基函数的有界性和Lipschitz连续性。我们证明了有限时间精确恢复是高概率的，即使当$p$接近1时。不同于以往的工作，主要集中在独立同分布(I.I.D.)并且只为系统学习提供渐近保证，这项研究首次在高度一般的扰动模型下对非线性动态系统进行了有限时间分析。我们的框架允许扰动中可能的时间相关性，并适应半不经意的对抗性攻击，极大地拓宽了现有理论结果的范围。



## **27. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

努力图表：量化人工智能使用风险以进行漏洞评估 cs.CR

8 pages; accepted for the 16th International Conference on Cloud  Computing, GRIDs, and Virtualization (Cloud Computing 2025), Valencia, Spain,  2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16392v1) [paper-pdf](http://arxiv.org/pdf/2503.16392v1)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.

摘要: 随着基于人工智能的软件变得广泛可用，利用其高自动化和复杂模式识别等能力的风险可能会显著增加。用来攻击非人工智能资产的人工智能被称为攻击性人工智能。目前的研究探索了如何利用攻击性人工智能，以及如何对其使用进行分类。此外，正在为组织内基于人工智能的资产开发威胁建模方法。然而，还有一些差距需要填补。首先，有必要量化造成人工智能威胁的因素。其次，需要创建威胁模型来分析被人工智能攻击的风险，以便对组织的所有资产进行漏洞评估。这在云环境中尤为关键和具有挑战性，因为在云环境中，复杂的基础设施和访问控制环境非常普遍。量化和进一步分析攻击性人工智能构成的威胁的能力使分析师能够对漏洞进行排名，并确定实施主动对策的优先顺序。为了解决这些差距，本文引入了努力图，这是一种直观、灵活和有效的威胁建模方法，用于分析使用攻击性人工智能进行对手漏洞利用所需的努力。虽然威胁模型是功能性的，并提供了有价值的支持，但其设计选择需要在未来的工作中进一步进行经验验证。



## **28. RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility in Autonomous Vehicles**

RECFL：一个不确定性感知框架，通过平衡自动驾驶汽车中的隐私、公平和实用性，实现负责任的联邦学习 cs.LG

Submitted to PETS 2025 (under review)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16251v1) [paper-pdf](http://arxiv.org/pdf/2503.16251v1)

**Authors**: Dawood Wasif, Terrence J. Moore, Jin-Hee Cho

**Abstract**: Autonomous vehicles (AVs) increasingly rely on Federated Learning (FL) to enhance perception models while preserving privacy. However, existing FL frameworks struggle to balance privacy, fairness, and robustness, leading to performance disparities across demographic groups. Privacy-preserving techniques like differential privacy mitigate data leakage risks but worsen fairness by restricting access to sensitive attributes needed for bias correction. This work explores the trade-off between privacy and fairness in FL-based object detection for AVs and introduces RESFL, an integrated solution optimizing both. RESFL incorporates adversarial privacy disentanglement and uncertainty-guided fairness-aware aggregation. The adversarial component uses a gradient reversal layer to remove sensitive attributes, reducing privacy risks while maintaining fairness. The uncertainty-aware aggregation employs an evidential neural network to weight client updates adaptively, prioritizing contributions with lower fairness disparities and higher confidence. This ensures robust and equitable FL model updates. We evaluate RESFL on the FACET dataset and CARLA simulator, assessing accuracy, fairness, privacy resilience, and robustness under varying conditions. RESFL improves detection accuracy, reduces fairness disparities, and lowers privacy attack success rates while demonstrating superior robustness to adversarial conditions compared to other approaches.

摘要: 自动驾驶汽车(AVs)越来越依赖联邦学习(FL)来增强感知模型，同时保护隐私。然而，现有的FL框架难以平衡隐私、公平和健壮性，导致不同人口群体之间的表现差异。像差异隐私这样的隐私保护技术降低了数据泄露风险，但限制了对偏见纠正所需的敏感属性的访问，从而恶化了公平性。该工作探讨了基于FL的AVS目标检测中的隐私和公平性之间的权衡，并介绍了一种优化两者的集成解决方案RESFL。RESFL融合了对抗性隐私解缠和不确定性引导的公平感知聚合。对抗性组件使用梯度反转层来移除敏感属性，从而在保持公平性的同时降低隐私风险。不确定性感知聚合使用证据神经网络自适应地对客户端更新进行加权，以较低的公平性差异和较高的置信度对贡献进行优先排序。这确保了稳健和公平的FL模型更新。我们在刻面数据集和CALA模拟器上对RESFL进行了评估，评估了在不同条件下的准确性、公平性、隐私弹性和健壮性。与其他方法相比，RESFL提高了检测精度，减少了公平性差异，降低了隐私攻击成功率，同时表现出对敌对条件的卓越稳健性。



## **29. AI Agents in Cryptoland: Practical Attacks and No Silver Bullet**

加密土地中的人工智能代理：实际攻击和没有银弹 cs.CR

12 pages, 8 figures

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16248v1) [paper-pdf](http://arxiv.org/pdf/2503.16248v1)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness, yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating. Our findings indicate that prompt-based defenses are insufficient, as malicious inputs can corrupt an agent's stored context, creating cascading vulnerabilities across interactions and platforms. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible.

摘要: 人工智能代理与Web3生态系统的集成利用了它们在自治和开放方面的互补潜力，但也带来了未被探索的安全风险，因为这些代理与金融协议和一成不变的智能合同动态交互。本文研究了基于区块链的金融生态系统中人工智能代理在现实场景中面临对抗性威胁时的脆弱性。我们引入了上下文操纵的概念--一种利用未受保护的上下文面的综合攻击载体，包括输入通道、内存模块和外部数据馈送。通过对Elizabeth OS的经验分析，我们展示了攻击者如何通过向提示或历史交互记录中注入恶意指令来操纵上下文，从而导致意外的资产转移和协议违规，这可能是经济上的毁灭性破坏。我们的发现表明，基于提示的防御是不够的，因为恶意输入可以破坏代理的存储上下文，从而在交互和平台之间产生级联漏洞。这项研究突显了开发既安全又可信的人工智能代理的迫切需要。



## **30. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **31. 2DSig-Detect: a semi-supervised framework for anomaly detection on image data using 2D-signatures**

2DSig-Detect：使用2D签名对图像数据进行异常检测的半监督框架 cs.CV

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.04982v2) [paper-pdf](http://arxiv.org/pdf/2409.04982v2)

**Authors**: Xinheng Xie, Kureha Yamaguchi, Margaux Leblanc, Simon Malzard, Varun Chhabra, Victoria Nockles, Yue Wu

**Abstract**: The rapid advancement of machine learning technologies raises questions about the security of machine learning models, with respect to both training-time (poisoning) and test-time (evasion, impersonation, and inversion) attacks. Models performing image-related tasks, e.g. detection, and classification, are vulnerable to adversarial attacks that can degrade their performance and produce undesirable outcomes. This paper introduces a novel technique for anomaly detection in images called 2DSig-Detect, which uses a 2D-signature-embedded semi-supervised framework rooted in rough path theory. We demonstrate our method in adversarial settings for training-time and test-time attacks, and benchmark our framework against other state of the art methods. Using 2DSig-Detect for anomaly detection, we show both superior performance and a reduction in the computation time to detect the presence of adversarial perturbations in images.

摘要: 机器学习技术的快速发展引发了有关机器学习模型安全性的问题，包括训练时（中毒）和测试时（逃避、模仿和倒置）攻击。执行图像相关任务（例如检测和分类）的模型很容易受到对抗攻击，这些攻击可能会降低其性能并产生不良结果。本文介绍了一种名为2DSig-Detect的图像异常检测新技术，该技术使用植根于粗糙路径理论的2D签名嵌入半监督框架。我们在训练时和测试时攻击的对抗环境中展示了我们的方法，并将我们的框架与其他最先进的方法进行基准测试。使用2DSig-Detect进行异常检测，我们既表现出卓越的性能，又减少了检测图像中存在对抗性扰动的计算时间。



## **32. REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models**

REVAR：大型视觉语言模型可靠性和价值的理解评估 cs.CV

45 pages, 5 figures, 18 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16566v1) [paper-pdf](http://arxiv.org/pdf/2503.16566v1)

**Authors**: Jie Zhang, Zheng Yuan, Zhongqi Wang, Bei Yan, Sibo Wang, Xiangkui Cao, Zonghui Guo, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the \textbf{RE}liability and \textbf{VAL}ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (\eg, perceptual accuracy and hallucination tendencies) and robustness (\eg, resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (\eg, bias and moral understanding), safety issues (\eg, toxicity and jailbreak vulnerabilities), and privacy problems (\eg, privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.

摘要: 大型视觉语言模型(LVLM)的快速发展突显了从不同维度评估这些模型的综合评估框架的必要性。虽然现有的基准侧重于感知能力、认知能力和对抗对手攻击的安全性等特定方面，但它们往往缺乏提供对LVLMS的优势和局限性的全面了解所需的广度和深度。为了弥补这一差距，我们引入了REVAL，这是一个全面的基准，旨在评估LVLM的责任和价值。REVAL包括144K多个图文视觉问答(VQA)样本，分为两个主要部分：可靠性，评估真实性(例如，感知准确性和幻觉倾向)和稳健性(例如，对对手攻击、排版攻击和图像损坏的恢复能力)；价值观，评估伦理问题(例如，偏见和道德理解)、安全问题(例如，毒性和越狱漏洞)和隐私问题(例如，隐私意识和隐私泄露)。我们评估了26款机型，包括主流开源LVLMS和著名的闭源机型，如GPT-4o和Gemini-1.5-Pro。我们的发现表明，尽管目前的LVLM在感知任务和毒性避免方面表现出色，但它们在对抗场景、隐私保护和伦理推理方面表现出显著的脆弱性。这些见解强调了未来改进的关键领域，指导开发更安全、可靠和符合道德规范的LVLM。REVAL为研究人员提供了一个强大的框架来系统地评估和比较LVLM，促进了该领域的进步。



## **33. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

SAUCE：使用稀疏自动编码器的视觉语言模型中的选择性概念消除 cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

摘要: 视觉语言模型(VLM)的遗忘方法主要采用来自大型语言模型(LLM)的技术，依赖于需要大量注释遗忘集的权重更新。此外，这些方法在粗粒度上执行遗忘，经常导致过度遗忘和降低模型效用。为了解决这个问题，我们引入了SASE，这是一种新的方法，它利用稀疏自动编码器(SAE)在VLM中进行细粒度和选择性的概念遗忘。简而言之，SASE首先训练SAE捕获高维的、语义丰富的稀疏特征。然后确定与目标概念最相关的特征以进行遗忘。在推理过程中，它有选择地修改这些特征以抑制特定概念，同时保留不相关的信息。我们在两个不同的VLM，LLaVA-v1.5-7B和Llama-3.2-11B-Vision-Indict上评估SAUE，跨越两种类型的任务：具体概念遗忘(物体和运动场景)和抽象概念遗忘(情绪、颜色和材料)，总共包含60个概念。大量的实验表明，在保持可比的模型效用的情况下，SASE在遗忘质量方面比最先进的方法高出18.04%。此外，我们还研究了SASE对广泛使用的敌意攻击的健壮性、其跨模型的可转移性以及其在处理多个并发遗忘请求时的可扩展性。我们的研究结果表明，SASE是一种有效且可扩展的解决方案，可用于解决VLMS中的选择性概念遗忘问题。



## **34. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

DroidTTP：使用TTP映射Android应用程序以实现网络威胁情报 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.

摘要: Android设备广泛用于银行和通信等敏感操作，使其成为网络威胁的主要目标，特别是高级持久性威胁(APT)和复杂的恶意软件攻击。传统的恶意软件检测方法依赖于二进制分类，无法提供对敌对战术、技术和过程(TTP)的洞察。了解恶意软件行为对于加强网络安全防御至关重要。为了弥补这一差距，我们在MITRE ATT&CK框架的基础上引入了DroidTTP，一个将Android恶意软件行为映射到TTP的框架。我们精心挑选的数据集明确地将MITRE TTP链接到Android应用程序。我们开发了一个自动化解决方案，利用问题转换方法(PTA)和大型语言模型(LLM)将应用程序映射到战术和技术。此外，我们使用了具有即时工程和LLM微调的检索-增强生成(RAG)来进行TTP预测。我们的结构化流程包括数据集创建、超参数调整、数据增强、特征选择、模型开发和基于Shap的模型可解释性。在LLMS中，大羊驼在战术分类上表现最好，贾卡德相似度为0.9583，Hamming损失为0.0182；在技术分类上表现最好，Jaccard相似度为0.9348，Hamming损失为0.0127。然而，标签Powerset XGBoost模型的表现优于LLMS，战术分类的Jaccard相似度为0.9893，技术分类的Jaccard相似度为0.9753，Hamming损失分别为0.0054和0.0050。虽然XGBoost表现出了优越的性能，但狭窄的差距突显了基于LLM的方法在TTP分类中的潜力。



## **35. Cyber Threats in Financial Transactions -- Addressing the Dual Challenge of AI and Quantum Computing**

金融交易中的网络威胁--应对人工智能和量子计算的双重挑战 cs.CR

38 Pages, 3 tables, Technical Report,  https://www.acfti.org/cftirc-community/technical-report-1-quantum-finance-cyber-threats

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.15678v1) [paper-pdf](http://arxiv.org/pdf/2503.15678v1)

**Authors**: Ahmed M. Elmisery, Mirela Sertovic, Andrew Zayin, Paul Watson

**Abstract**: The financial sector faces escalating cyber threats amplified by artificial intelligence (AI) and the advent of quantum computing. AI is being weaponized for sophisticated attacks like deepfakes and AI-driven malware, while quantum computing threatens to render current encryption methods obsolete. This report analyzes these threats, relevant frameworks, and possible countermeasures like quantum cryptography. AI enhances social engineering and phishing attacks via personalized content, lowers entry barriers for cybercriminals, and introduces risks like data poisoning and adversarial AI. Quantum computing, particularly Shor's algorithm, poses a fundamental threat to current encryption standards (RSA and ECC), with estimates suggesting cryptographically relevant quantum computers could emerge within the next 5-30 years. The "harvest now, decrypt later" scenario highlights the urgency of transitioning to quantum-resistant cryptography. This is key. Existing legal frameworks are evolving to address AI in cybercrime, but quantum threats require new initiatives. International cooperation and harmonized regulations are crucial. Quantum Key Distribution (QKD) offers theoretical security but faces practical limitations. Post-quantum cryptography (PQC) is a promising alternative, with ongoing standardization efforts. Recommendations for international regulators include fostering collaboration and information sharing, establishing global standards, supporting research and development in quantum security, harmonizing legal frameworks, promoting cryptographic agility, and raising awareness and education. The financial industry must adopt a proactive and adaptive approach to cybersecurity, investing in research, developing migration plans for quantum-resistant cryptography, and embracing a multi-faceted, collaborative strategy to build a resilient, quantum-safe, and AI-resilient financial ecosystem

摘要: 金融行业面临不断升级的网络威胁，人工智能(AI)和量子计算的出现放大了这一威胁。人工智能正被武器化，用于深度假冒和人工智能驱动的恶意软件等复杂攻击，而量子计算可能会使当前的加密方法过时。这份报告分析了这些威胁、相关框架和可能的对策，如量子密码学。人工智能通过个性化内容加强社会工程和网络钓鱼攻击，降低网络犯罪分子的进入门槛，并引入数据中毒和对抗性人工智能等风险。量子计算，特别是肖尔的算法，对当前的加密标准(RSA和ECC)构成了根本威胁，估计表明，在未来5-30年内，可能会出现与密码相关的量子计算机。“现在收获，以后解密”的情景凸显了向量子抵抗密码术过渡的紧迫性。这是关键。现有的法律框架正在演变，以解决网络犯罪中的人工智能问题，但量子威胁需要新的举措。国际合作和统一的条例至关重要。量子密钥分发(QKD)提供了理论上的安全性，但也面临着实践上的局限性。随着标准化工作的进行，后量子密码学(PQC)是一种很有前途的替代方案。对国际监管机构的建议包括促进合作和信息共享，建立全球标准，支持量子安全方面的研究和开发，协调法律框架，促进加密灵活性，以及提高认识和教育。金融行业必须对网络安全采取主动和自适应的方法，投资于研究，制定量子耐加密的迁移计划，并采用多方面的协作战略，以构建具有弹性、量子安全和人工智能弹性的金融生态系统



## **36. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

不，我当然可以！使用无害微调数据可以利用拒绝机制 cs.CR

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.19537v2) [paper-pdf](http://arxiv.org/pdf/2502.19537v2)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Krishnamurthy Dvijotham

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.

摘要: 领先的语言模型(LM)提供商，如OpenAI和Google，提供了微调的API，允许客户根据特定的用例调整LMS。为防止误用，这些LM提供程序实施过滤机制以阻止有害的微调数据。因此，试图通过这些API生成不安全的LMS的攻击者必须创建不能识别有害的对抗性训练数据。我们在这方面做了三点贡献：1.我们证明了许多现有的使用无害数据来创建不安全的LMS的攻击依赖于消除其响应的前几个令牌中的模型拒绝。2.我们证明了这样的先前攻击可以通过一个简单的防御来阻止，即预先填充来自对齐模型的前几个令牌，然后让微调的模型填充其余的令牌。3.我们描述了一种新的数据中毒攻击，``不，我当然可以执行‘’(Noice)，它利用LM的公式化拒绝机制来引发有害的响应。通过训练LM在满足良性请求之前出于安全考虑拒绝这些请求，我们能够越狱几个开源模型和一个封闭源代码模型(GPT-40)。我们对GPT-40的攻击成功率(ASR)为57%；我们的攻击从OpenAI获得了错误赏金。相对于由简单防御保护的开源模型，我们将ASR平均提高了3.25倍，而之前仅使用无害数据的攻击性能最好。Noice展示了重复拒绝机制的可利用性，并拓宽了对封闭源代码模型面临的无害数据威胁的理解。



## **37. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **38. Adaptive Pruning with Module Robustness Sensitivity: Balancing Compression and Robustness**

具有模块稳健性敏感性的自适应修剪：平衡压缩和稳健性 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.15176v2) [paper-pdf](http://arxiv.org/pdf/2410.15176v2)

**Authors**: Lincen Bai, Hedi Tabia, Raúl Santos-Rodríguez

**Abstract**: Neural network pruning has traditionally focused on weight-based criteria to achieve model compression, frequently overlooking the crucial balance between adversarial robustness and accuracy. Existing approaches often fail to preserve robustness in pruned networks, leaving them more susceptible to adversarial attacks. This paper introduces Module Robustness Sensitivity (MRS), a novel metric that quantifies layer-wise sensitivity to adversarial perturbations and dynamically informs pruning decisions. Leveraging MRS, we propose Module Robust Pruning and Fine-Tuning (MRPF), an adaptive pruning algorithm compatible with any adversarial training method, offering both flexibility and scalability. Extensive experiments on SVHN, CIFAR, and Tiny-ImageNet across diverse architectures, including ResNet, VGG, and MobileViT, demonstrate that MRPF significantly enhances adversarial robustness while maintaining competitive accuracy and computational efficiency. Furthermore, MRPF consistently outperforms state-of-the-art structured pruning methods in balancing robustness, accuracy, and compression. This work establishes a practical and generalizable framework for robust pruning, addressing the long-standing trade-off between model compression and robustness preservation.

摘要: 传统上，神经网络剪枝侧重于基于权重的标准来实现模型压缩，但往往忽略了对手健壮性和准确性之间的关键平衡。现有的方法往往不能在经过剪枝的网络中保持健壮性，从而使它们更容易受到对手攻击。本文引入了模块健壮性敏感度(MRS)，这是一种新的度量，它量化了对对手扰动的层级敏感度，并动态地通知剪枝决策。利用MRS，我们提出了模块稳健剪枝和精调(MRPF)，这是一种与任何对抗性训练方法兼容的自适应剪枝算法，提供了灵活性和可扩展性。在包括ResNet、VGG和MobileViT在内的不同体系结构上对SVHN、CIFAR和Tiny-ImageNet进行的大量实验表明，MRPF在保持竞争精度和计算效率的同时，显著增强了对手的健壮性。此外，在稳健性、准确性和压缩方面，MRPF始终优于最先进的结构化剪枝方法。这项工作建立了一个实用和可推广的稳健剪枝框架，解决了模型压缩和稳健性保持之间的长期权衡。



## **39. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

通过动态最大化优化改进普遍对抗扰动的推广 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.12793v2) [paper-pdf](http://arxiv.org/pdf/2503.12793v2)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.

摘要: 深度神经网络(DNN)容易受到普遍的对抗性扰动(UAP)的影响。这些扰动是精心设计的，目的是在所有样本类中普遍欺骗目标模型。与实例特定的对抗性示例(AE)不同，生成UAP更加复杂，因为它们必须在广泛的数据样本和模型中推广。我们的研究表明，现有的通用攻击方法使用带有静态模型参数快照的DNN来优化UAP，没有充分利用DNN的潜力来生成更有效的UAP。我们建议使用动态模型-数据对来生成UAP，而不是针对具有固定训练集的静态DNN模型来优化UAP。特别是，我们引入了动态最大优化策略，旨在对UAP进行各种最优模型-数据对的优化。我们称这种方法为DM-UAP。DM-UAP使用迭代的最大-最小-最小优化框架来细化模型-数据对，并结合课程UAP学习算法来彻底检查模型参数和数据的组合空间。在ImageNet数据集上的综合实验表明，DM-UAP显著增强了UAP的跨样本普适性和跨模型可转移性。仅使用500个样本生成UAP，DM-UAP的性能优于最先进的方法，平均傻瓜率提高了12.108%。



## **40. Robustness bounds on the successful adversarial examples in probabilistic models: Implications from Gaussian processes**

概率模型中成功对抗示例的鲁棒性界限：高斯过程的含义 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2403.01896v2) [paper-pdf](http://arxiv.org/pdf/2403.01896v2)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification, a probabilistic inference model. We proved a new upper bound of the probability of a successful AE attack that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.

摘要: 对抗示例（AE）是机器学习的一种攻击方法，通过向数据添加难以察觉的扰动来制作，从而导致错误分类。在本文中，我们基于高斯过程（GP）分类（一种概率推理模型）研究了成功AE的概率的上限。我们证明了AE攻击成功概率的新上界，该上界取决于AE的扰动规范、GP中使用的核函数以及训练数据集中具有不同标签的最接近对的距离。令人惊讶的是，无论样本数据集的分布如何，上限都会确定。我们表明我们的理论结果通过使用ImageNet的实验得到了证实。此外，我们还表明，改变核函数的参数会导致AE成功概率上界的变化。



## **41. A Semantic and Clean-label Backdoor Attack against Graph Convolutional Networks**

针对图卷积网络的语义和干净标签后门攻击 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14922v1) [paper-pdf](http://arxiv.org/pdf/2503.14922v1)

**Authors**: Jiazhu Dai, Haoyu Sun

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in graph-structured tasks such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called the backdoor attack, where the adversary can inject a hidden backdoor into the GCNs so that the backdoored model performs well on benign samples, whereas its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. Clean-label backdoor attack and semantic backdoor attack are two new backdoor attacks to Deep Neural Networks (DNNs), they are more imperceptible and have posed new and serious threats. The semantic and clean-label backdoor attack is not fully explored in GCNs. In this paper, we propose a semantic and clean-label backdoor attack against GCNs under the context of graph classification to reveal the existence of this security vulnerability in GCNs. Specifically, SCLBA conducts an importance analysis on graph samples to select one type of node as semantic trigger, which is then inserted into the graph samples to create poisoning samples without changing the labels of the poisoning samples to the attacker-specified target label. We evaluate SCLBA on multiple datasets and the results show that SCLBA can achieve attack success rates close to 99% with poisoning rates of less than 3%, and with almost no impact on the performance of model on benign samples.

摘要: 图卷积网络(GCNS)在节点分类、图分类等图结构任务中表现出优异的性能。然而，最近的研究表明，GCNS容易受到一种称为后门攻击的新型威胁的攻击，在这种威胁中，攻击者可以向GCNS注入隐藏的后门，以便后门模型在良性样本上执行得很好，而如果隐藏的后门被攻击者定义的触发器激活，则其预测将被恶意更改为攻击者指定的目标标签。干净标签后门攻击和语义后门攻击是深度神经网络的两种新的后门攻击，它们的隐蔽性更强，已经构成了新的严重威胁。语义和干净标签的后门攻击在GCNS中没有得到充分的探索。为了揭示GCNS中存在的安全漏洞，提出了一种基于图分类的语义和干净标签的GCNS后门攻击方法。具体地说，SCLBA对图样本进行重要性分析，选择一种类型的节点作为语义触发器，然后将其插入到图样本中创建中毒样本，而不会将中毒样本的标签更改为攻击者指定的目标标签。我们在多个数据集上对SCLBA进行了评估，结果表明，SCLBA可以达到接近99%的攻击成功率，而投毒率低于3%，并且对良性样本的模型性能几乎没有影响。



## **42. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

ICLR 2025, fix typos in the proof

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2408.00315v4) [paper-pdf](http://arxiv.org/pdf/2408.00315v4)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **43. Synthesizing Grid Data with Cyber Resilience and Privacy Guarantees**

综合具有网络弹性和隐私保证的网格数据 eess.SY

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14877v1) [paper-pdf](http://arxiv.org/pdf/2503.14877v1)

**Authors**: Shengyang Wu, Vladimir Dvorkin

**Abstract**: Differential privacy (DP) provides a principled approach to synthesizing data (e.g., loads) from real-world power systems while limiting the exposure of sensitive information. However, adversaries may exploit synthetic data to calibrate cyberattacks on the source grids. To control these risks, we propose new DP algorithms for synthesizing data that provide the source grids with both cyber resilience and privacy guarantees. The algorithms incorporate both normal operation and attack optimization models to balance the fidelity of synthesized data and cyber resilience. The resulting post-processing optimization is reformulated as a robust optimization problem, which is compatible with the exponential mechanism of DP to moderate its computational burden.

摘要: 差异隐私（DP）提供了一种有原则的方法来合成数据（例如，负载）来自现实世界的电力系统，同时限制敏感信息的暴露。然而，对手可能会利用合成数据来校准对源网格的网络攻击。为了控制这些风险，我们提出了新的DP算法来合成数据，为源网格提供网络弹性和隐私保证。这些算法结合了正常操作和攻击优化模型，以平衡合成数据的保真度和网络弹性。所得的后处理优化被重新表述为鲁棒优化问题，该问题与DP的指数机制兼容，以减轻其计算负担。



## **44. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

时间上下文感知：针对大型语言模型多轮操纵攻击的防御框架 cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **45. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

多轮对话中社会工程的个性化攻击-- LLM模拟和检测代理 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大语言模型(LLM)驱动的聊天机器人，构成了社交媒体平台上的社交工程(SE)攻击的巨大风险。基于聊天的多轮交互中的SE检测比单实例检测要复杂得多，这是因为这些对话的动态性质。缓解这一威胁的一个关键因素是了解SE攻击的运作机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何导致他们的易感性。在这项工作中，我们提出了一个LLM代理框架SE-VSim，通过生成多话轮会话来模拟SE攻击机制。我们对具有不同个性特征的受害者代理进行建模，以评估心理特征如何影响操纵的易感性。使用1000多个模拟对话的数据集，我们检查了攻击场景，在这些场景中，伪装成招聘者、资助机构和记者的对手试图提取敏感信息。基于这一分析，我们提出了一个概念证明，SE-OmniGuard，通过利用受害者个性的先验知识，评估攻击策略，并监控对话中的信息交换来识别潜在的SE尝试，为用户提供个性化保护。



## **46. Adversarial Robustness in Parameter-Space Classifiers**

参数空间分类器中的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.20314v2) [paper-pdf](http://arxiv.org/pdf/2502.20314v2)

**Authors**: Tamir Shor, Ethan Fetaya, Chaim Baskin, Alex Bronstein

**Abstract**: Implicit Neural Representations (INRs) have been recently garnering increasing interest in various research fields, mainly due to their ability to represent large, complex data in a compact and continuous manner. Past work further showed that numerous popular downstream tasks can be performed directly in the INR parameter-space. Doing so can substantially reduce the computational resources required to process the represented data in their native domain. A major difficulty in using modern machine-learning approaches, is their high susceptibility to adversarial attacks, which have been shown to greatly limit the reliability and applicability of such methods in a wide range of settings. In this work, we show that parameter-space models trained for classification are inherently robust to adversarial attacks -- without the need of any robust training. To support our claims, we develop a novel suite of adversarial attacks targeting parameter-space classifiers, and furthermore analyze practical considerations of attacking parameter-space classifiers.

摘要: 隐式神经表示(INR)最近在各个研究领域引起了越来越多的兴趣，这主要是因为它们能够以紧凑和连续的方式表示大型、复杂的数据。过去的工作进一步表明，许多流行的下游任务可以直接在INR参数空间中执行。这样做可以大大减少在其本地域中处理所表示的数据所需的计算资源。使用现代机器学习方法的一个主要困难是它们对对手攻击的高度敏感性，这已被证明在广泛的环境中极大地限制了这种方法的可靠性和适用性。在这项工作中，我们证明了为分类而训练的参数空间模型在本质上对对手攻击是健壮的--而不需要任何健壮的训练。为了支持我们的观点，我们开发了一套新的针对参数空间分类器的对抗性攻击，并进一步分析了攻击参数空间分类器的实际考虑。



## **47. Anomaly-Flow: A Multi-domain Federated Generative Adversarial Network for Distributed Denial-of-Service Detection**

异常流：一种用于分布式拒绝服务检测的多域联邦生成对抗网络 cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14618v1) [paper-pdf](http://arxiv.org/pdf/2503.14618v1)

**Authors**: Leonardo Henrique de Melo, Gustavo de Carvalho Bertoli, Michele Nogueira, Aldri Luiz dos Santos, Lourenço Alves Pereira Junior

**Abstract**: Distributed denial-of-service (DDoS) attacks remain a critical threat to Internet services, causing costly disruptions. While machine learning (ML) has shown promise in DDoS detection, current solutions struggle with multi-domain environments where attacks must be detected across heterogeneous networks and organizational boundaries. This limitation severely impacts the practical deployment of ML-based defenses in real-world settings.   This paper introduces Anomaly-Flow, a novel framework that addresses this critical gap by combining Federated Learning (FL) with Generative Adversarial Networks (GANs) for privacy-preserving, multi-domain DDoS detection. Our proposal enables collaborative learning across diverse network domains while preserving data privacy through synthetic flow generation. Through extensive evaluation across three distinct network datasets, Anomaly-Flow achieves an average F1-score of $0.747$, outperforming baseline models. Importantly, our framework enables organizations to share attack detection capabilities without exposing sensitive network data, making it particularly valuable for critical infrastructure and privacy-sensitive sectors.   Beyond immediate technical contributions, this work provides insights into the challenges and opportunities in multi-domain DDoS detection, establishing a foundation for future research in collaborative network defense systems. Our findings have important implications for academic research and industry practitioners working to deploy practical ML-based security solutions.

摘要: 分布式拒绝服务(DDoS)攻击仍然是对互联网服务的严重威胁，造成代价高昂的中断。虽然机器学习(ML)在DDoS检测方面显示出了希望，但当前的解决方案在多域环境中苦苦挣扎，在多域环境中，必须跨不同的网络和组织边界检测攻击。这一限制严重影响了基于ML的防御在现实世界环境中的实际部署。本文介绍了一种新的框架，它通过将联邦学习(FL)和生成性对抗网络(GANS)相结合来解决这一关键缺陷，以实现隐私保护的多域DDoS检测。我们的建议支持跨不同网络领域的协作学习，同时通过合成流量生成保护数据隐私。通过对三个不同的网络数据集进行广泛的评估，异常流实现了0.747美元的F1平均得分，表现优于基线模型。重要的是，我们的框架使组织能够在不暴露敏感网络数据的情况下共享攻击检测功能，使其对关键基础设施和隐私敏感部门特别有价值。除了直接的技术贡献外，这项工作还提供了对多域DDoS检测的挑战和机遇的见解，为未来协作网络防御系统的研究奠定了基础。我们的发现对致力于部署实用的基于ML的安全解决方案的学术研究和行业从业者具有重要的意义。



## **48. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

VGFL-SA：基于对比学习的垂直图联邦学习结构攻击 cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.16793v2) [paper-pdf](http://arxiv.org/pdf/2502.16793v2)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.

摘要: 图形神经网络(GNN)因其从图形数据中学习表示的能力而受到关注。由于隐私问题和利益冲突阻碍了客户之间直接共享图形数据，垂直图形联合学习(VGFL)框架已经开发出来。最近的研究表明，VGFL很容易受到降低性能的对抗性攻击。然而，在VGFL领域中，一个常见的问题是客户端节点通常是未标记的。因此，现有的攻击依赖于标记信息的可用性来获得梯度，其适用性受到固有的限制。这一限制排除了它们在实际、真实环境中的部署。针对上述问题，我们提出了一种新的针对VGFL的图对抗攻击，称为VGFL-SA，通过修改本地客户端结构而不使用标签来降低VGFL的性能。具体地说，VGFL-SA使用对比学习方法在本地客户端训练之前完成攻击。VGFL-SA首先获取中毒客户端的图结构和节点特征信息，然后通过基于节点度的边增强和特征置乱增强生成对比视图。然后，VGFL-SA使用共享图编码器得到每个视点的嵌入，并通过对比函数得到邻接矩阵的梯度。最后，使用梯度修正规则生成扰动边缘。我们通过在真实数据集上执行节点分类任务来验证VGFL-SA的性能，结果表明VGFL-SA具有良好的攻击有效性和可转移性。



## **49. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

揭示随机化在多类对抗分类中的作用：来自图论的见解 cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14299v1) [paper-pdf](http://arxiv.org/pdf/2503.14299v1)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.

摘要: 随机化作为提高机器学习模型对抗性稳健性的一种手段，最近引起了人们的广泛关注。不幸的是，到目前为止，许多理论分析都集中在二进制分类上，对更复杂的多类设置只提供了有限的见解。在本文中，我们从图论领域汲取灵感，朝着缩小这一差距迈出了一步。我们的分析集中在离散数据分布上，允许我们在集合打包问题的良好框架内求解对抗性风险最小化问题。通过这样做，我们能够确定数据分布支持上的三个结构条件，这三个条件是随机化提高稳健性所必需的。此外，我们能够构建几个数据分布，其中(与二进制分类相反)从确定性解决方案切换到随机解决方案显著地降低了最优对抗风险。这些发现突显了随机化在增强多类分类中对抗攻击的稳健性方面所起的关键作用。



## **50. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

Under review

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2405.18770v2) [paper-pdf](http://arxiv.org/pdf/2405.18770v2)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. Our experiments show that MAT can effectively be applied to different VL models and tasks to improve adversarial robustness, outperforming previous efforts. Our code will be made public upon acceptance.

摘要: 预先训练好的视觉语言(VL)模型极易受到敌意攻击。然而，现有的防御方法主要集中在图像分类上，忽略了VL任务的两个关键方面：多模式攻击，其中图像和文本都可以被干扰，以及图像和文本的一对多关系，其中一幅图像可以对应于多个文本描述，反之亦然(1：N和N：1)。该工作首次探索了VL任务中针对多模式攻击的防御策略，而以往的VL防御方法侧重于视觉稳健性。我们提出了多模式对抗训练(MAT)，它在训练过程中结合了图像和文本模式的对抗扰动，显著优于现有的单模式防御。此外，我们发现MAT受到VL训练数据中确定性的一对一(1：1)图文对的限制。为了解决这个问题，我们进行了一项关于利用一对多关系来增强健壮性的全面研究，调查了各种增强技术。我们的分析表明，为了更有效地防御，增强的图文对应该很好地对齐、多样化，但又要避免分布偏移--这是以前的研究忽视的条件。我们的实验表明，MAT可以有效地应用于不同的虚拟学习模型和任务，以提高对手的健壮性，表现出优于以往的努力。我们的代码将在接受后公开。



