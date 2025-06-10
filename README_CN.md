# Latest Adversarial Attack Papers
**update at 2025-06-10 15:44:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

代码大型语言模型的对抗性攻击分类和鲁棒性测试 cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.

摘要: 大型语言模型（LLM）已成为代码生成、完成和分析等软件开发任务的重要工具。随着它们与工作流程集成的加深，确保针对漏洞（尤其是由多样化或敌对输入触发的漏洞）的鲁棒性变得越来越重要。当模型遇到受干扰的任务描述、代码或评论时，此类漏洞可能会导致不正确或不安全的代码生成。之前的研究经常忽视自然语言在指导代码任务中的作用。本研究调查了自然语言输入（包括提示、评论和描述）中的对抗性扰动如何影响LLM for Code（LLM4Code）。它检查字符、单词和句子层面上的干扰的影响，以识别最有影响力的漏洞。我们分析了多个项目（例如，ReCode、OpenAttack）和数据集（例如，HumanEval，MBPP），建立了对抗性攻击的分类。第一个维度对输入类型代码、提示或注释进行分类，而第二个维度重点关注粒度：字符、单词或业务级别的更改。我们采用了混合方法，将定量性能指标与定性漏洞分析相结合。LLM 4Code模型显示出不同扰动类型的鲁棒性不同。句子级别的攻击效果最差，这表明模型能够适应更广泛的背景变化。相比之下，词级扰动带来了严重的挑战，暴露了语义漏洞。初级效应各不相同，表明模型对微妙的语法偏差的敏感性。我们的研究提供了一个结构化框架来测试LLM 4 Code稳健性，并强调自然语言在对抗性评估中的关键作用。提高模型对语义级中断的弹性对于安全可靠的代码生成系统至关重要。



## **2. CAPAA: Classifier-Agnostic Projector-Based Adversarial Attack**

CAPPA：基于分类不可知投影仪的对抗攻击 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.00978v2) [paper-pdf](http://arxiv.org/pdf/2506.00978v2)

**Authors**: Zhan Li, Mingyu Zhao, Xin Dong, Haibin Ling, Bingyao Huang

**Abstract**: Projector-based adversarial attack aims to project carefully designed light patterns (i.e., adversarial projections) onto scenes to deceive deep image classifiers. It has potential applications in privacy protection and the development of more robust classifiers. However, existing approaches primarily focus on individual classifiers and fixed camera poses, often neglecting the complexities of multi-classifier systems and scenarios with varying camera poses. This limitation reduces their effectiveness when introducing new classifiers or camera poses. In this paper, we introduce Classifier-Agnostic Projector-Based Adversarial Attack (CAPAA) to address these issues. First, we develop a novel classifier-agnostic adversarial loss and optimization framework that aggregates adversarial and stealthiness loss gradients from multiple classifiers. Then, we propose an attention-based gradient weighting mechanism that concentrates perturbations on regions of high classification activation, thereby improving the robustness of adversarial projections when applied to scenes with varying camera poses. Our extensive experimental evaluations demonstrate that CAPAA achieves both a higher attack success rate and greater stealthiness compared to existing baselines. Codes are available at: https://github.com/ZhanLiQxQ/CAPAA.

摘要: 基于投影仪的对抗攻击旨在投影精心设计的光图案（即，对抗性投影）到场景上以欺骗深层图像分类器。它在隐私保护和开发更强大的分类器方面具有潜在的应用。然而，现有的方法主要关注单个分类器和固定的相机姿态，通常忽视了多分类器系统和具有不同相机姿态的场景的复杂性。这种限制降低了引入新分类器或相机姿势时的有效性。在本文中，我们引入分类不可知投影仪的对抗攻击（CAPPA）来解决这些问题。首先，我们开发了一个新颖的分类器不可知的对抗性损失和优化框架，该框架聚合了来自多个分类器的对抗性损失和隐蔽性损失梯度。然后，我们提出了一种基于注意力的梯度加权机制，该机制将扰动集中在高分类激活区域，从而提高了应用于摄像机姿态变化的场景时对抗投影的鲁棒性。我们广泛的实验评估表明，CAPAA实现了更高的攻击成功率和更大的隐蔽性相比，现有的基线。代码可访问：https://github.com/ZhanLiQxQ/CAPAA。



## **3. Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability**

用保形预测增强对抗鲁棒性：保证模型可靠性的框架 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07804v1) [paper-pdf](http://arxiv.org/pdf/2506.07804v1)

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction.

摘要: 随着深度学习模型越来越多地部署在高风险应用中，针对对抗攻击的强大防御和可靠的性能保证变得至关重要。此外，仅靠准确性并不能为这些模型提供足够的保证或可靠的不确定性估计。这项研究通过利用保形预测的原则来推进对抗训练。具体来说，我们开发了一种对抗攻击方法，称为OPSA（最佳大小攻击），旨在通过在不要求覆盖保证的情况下最大化模型不确定性来降低任何重要性水平上的保形预测的效率。相应地，我们引入了OPSA-AT（对抗训练），这是一种将OPSA集成到新型适形训练范式中的防御策略。实验评估表明，与各种防御的基线方法相比，我们的OPSA攻击方法会引发更大的不确定性。相反，我们的OPSA-AT防御模型不仅显着增强了针对OPSA以及其他对抗攻击的鲁棒性，并保持了可靠的预测。我们的研究结果凸显了这种集成方法对于为安全关键领域开发值得信赖和有弹性的深度学习模型的有效性。我们的代码可在https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction上获取。



## **4. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

审判与信任：以全面的防御战略应对拜占庭袭击 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.07614v2) [paper-pdf](http://arxiv.org/pdf/2505.07614v2)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.

摘要: 机器学习的最新进展提高了性能，同时也增加了计算需求。虽然联邦和分布式设置可以解决这些问题，但其结构很容易受到恶意影响。在本文中，我们解决了一个特定的威胁，即拜占庭攻击，其中受影响的客户端注入对抗性更新以破坏全球融合。我们将信任分数概念与尝试函数方法相结合，以动态过滤离群值。我们的方法解决了以前方法的关键局限性，即使在拜占庭节点占多数时也允许功能。此外，我们的算法适用于Adam和RMSProp等广泛使用的缩放方法，以及实际场景，包括本地训练和部分参与。我们通过对从医疗机构收集的合成和真实心电图数据进行广泛的实验来验证我们方法的稳健性。此外，我们还对算法及其对上述实际设置的扩展进行了广泛的理论分析。我们方法的收敛保证与没有拜占庭干扰而开发的经典算法的收敛保证相当。



## **5. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07736v1) [paper-pdf](http://arxiv.org/pdf/2506.07736v1)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **6. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。这些风险可能会因最近的对抗性攻击、微调漏洞以及在高风险环境中越来越多地部署LLM而放大。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新的方法，从根本上破坏了LLM中有害行为的表示，提供了一种可扩展的解决方案来增强（潜在的固有）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **7. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



## **8. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

ProARD：渐进式对抗稳健性蒸馏：提供广泛的稳健学生 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07666v1) [paper-pdf](http://arxiv.org/pdf/2506.07666v1)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.

摘要: 对抗鲁棒性蒸馏（ARD）已成为增强轻量级深度神经网络抵御对抗攻击鲁棒性的有效方法。当前的ARD方法利用了一个强大的教师网络来培训一个强大的轻量级学生。然而，由于边缘设备的多样性和资源限制，当前的方法需要从头开始训练新的学生网络以满足特定的限制，从而导致巨大的计算成本和二氧化碳排放量增加。本文提出了渐进对抗鲁棒蒸馏（ProARD），可以对动态网络进行高效的一次性训练，该网络支持各种准确且稳健的学生网络，而无需再培训。我们首先基于动态层构建动态深度神经网络，通过涵盖每个设计阶段的宽度、深度和扩展的变化，以支持广泛的架构。然后，我们将规模最大的学生网络视为动态教师网络。ProARD使用权重共享机制训练这个动态网络，以联合优化动态教师网络及其内部学生网络。然而，由于计算动态网络中所有学生的精确梯度的计算成本很高，因此需要采样机制来选择学生的子集。我们表明，每次迭代中的随机学生抽样无法产生准确和稳健的学生。



## **9. Feature Statistics with Uncertainty Help Adversarial Robustness**

具有不确定性的特征统计有助于对抗稳健性 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2503.20583v2) [paper-pdf](http://arxiv.org/pdf/2503.20583v2)

**Authors**: Ran Wang, Xinlei Zhou, Meng Hu, Rihao Li, Wenhui Wu, Yuheng Jia

**Abstract**: Despite the remarkable success of deep neural networks (DNNs), the security threat of adversarial attacks poses a significant challenge to the reliability of DNNs. In this paper, both theoretically and empirically, we discover a universal phenomenon that has been neglected in previous works, i.e., adversarial attacks tend to shift the distributions of feature statistics. Motivated by this finding, and by leveraging the advantages of uncertainty-aware stochastic methods in building robust models efficiently, we propose an uncertainty-driven feature statistics adjustment module for robustness enhancement, named Feature Statistics with Uncertainty (FSU). It randomly resamples channel-wise feature means and standard deviations of examples from multivariate Gaussian distributions, which helps to reconstruct the perturbed examples and calibrate the shifted distributions. The calibration recovers some domain characteristics of the data for classification, thereby mitigating the influence of perturbations and weakening the ability of attacks to deceive models. The proposed FSU module has universal applicability in training, attacking, predicting, and fine-tuning, demonstrating impressive robustness enhancement ability at a trivial additional time cost. For example, by fine-tuning the well-established models with FSU, the state-of-the-art methods achieve up to 17.13% and 34.82% robustness improvement against powerful AA and CW attacks on benchmark datasets.

摘要: 尽管深度神经网络（DNN）取得了显着的成功，但对抗性攻击的安全威胁对DNN的可靠性构成了重大挑战。本文从理论上和经验上，我们发现了一个在以前的著作中被忽视的普遍现象，即对抗性攻击往往会改变特征统计数据的分布。受这一发现的启发，并利用不确定性感知随机方法在有效构建稳健模型方面的优势，我们提出了一种不确定性驱动的特征统计调整模块，用于增强稳健性，称为具有不确定性的特征统计（FSU）。它从多元高斯分布中随机重新采样示例的通道特征均值和标准差，这有助于重建受扰动的示例并校准移动的分布。校准恢复数据的一些领域特征进行分类，从而减轻扰动的影响并削弱攻击欺骗模型的能力。提出的FSU模块在训练、攻击、预测和微调方面具有普遍适用性，以微不足道的额外时间成本展示了令人印象深刻的鲁棒性增强能力。例如，通过使用FSU对成熟的模型进行微调，最先进的方法可以针对对基准数据集的强大AA和CW攻击实现高达17.13%和34.82%的稳健性改进。



## **10. RAID: A Dataset for Testing the Adversarial Robustness of AI-Generated Image Detectors**

RAGE：用于测试人工智能生成图像检测器对抗鲁棒性的数据集 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.03988v3) [paper-pdf](http://arxiv.org/pdf/2506.03988v3)

**Authors**: Hicham Eddoubi, Jonas Ricker, Federico Cocchi, Lorenzo Baraldi, Angelo Sotgiu, Maura Pintor, Marcella Cornia, Lorenzo Baraldi, Asja Fischer, Rita Cucchiara, Battista Biggio

**Abstract**: AI-generated images have reached a quality level at which humans are incapable of reliably distinguishing them from real images. To counteract the inherent risk of fraud and disinformation, the detection of AI-generated images is a pressing challenge and an active research topic. While many of the presented methods claim to achieve high detection accuracy, they are usually evaluated under idealized conditions. In particular, the adversarial robustness is often neglected, potentially due to a lack of awareness or the substantial effort required to conduct a comprehensive robustness analysis. In this work, we tackle this problem by providing a simpler means to assess the robustness of AI-generated image detectors. We present RAID (Robust evaluation of AI-generated image Detectors), a dataset of 72k diverse and highly transferable adversarial examples. The dataset is created by running attacks against an ensemble of seven state-of-the-art detectors and images generated by four different text-to-image models. Extensive experiments show that our methodology generates adversarial images that transfer with a high success rate to unseen detectors, which can be used to quickly provide an approximate yet still reliable estimate of a detector's adversarial robustness. Our findings indicate that current state-of-the-art AI-generated image detectors can be easily deceived by adversarial examples, highlighting the critical need for the development of more robust methods. We release our dataset at https://huggingface.co/datasets/aimagelab/RAID and evaluation code at https://github.com/pralab/RAID.

摘要: 人工智能生成的图像已经达到了人类无法可靠地将其与真实图像区分开来的质量水平。为了抵消欺诈和虚假信息的固有风险，检测人工智能生成的图像是一项紧迫的挑战和一个活跃的研究课题。虽然提出的许多方法声称可以实现高检测准确性，但它们通常是在理想化条件下进行评估的。特别是，对抗稳健性经常被忽视，这可能是由于缺乏意识或进行全面稳健性分析所需的大量努力。在这项工作中，我们通过提供一种更简单的方法来评估人工智能生成的图像检测器的稳健性来解决这个问题。我们介绍了RAIDs（人工智能生成图像检测器的稳健评估），这是一个包含72 k个多样化且高度可转移的对抗示例的数据集。该数据集是通过对七个最先进的检测器和由四种不同的文本到图像模型生成的图像进行攻击来创建的。大量实验表明，我们的方法可以生成对抗图像，这些图像以很高的成功率传输到未见的检测器，可以用于快速提供对检测器对抗鲁棒性的大致但仍然可靠的估计。我们的研究结果表明，当前最先进的人工智能生成图像检测器很容易被对抗性示例所欺骗，这凸显了开发更稳健方法的迫切需要。我们在https://huggingface.co/datasets/aimagelab/RAID上发布我们的数据集，并在https://github.com/pralab/RAID上发布评估代码。



## **11. Explore the vulnerability of black-box models via diffusion models**

通过扩散模型探索黑匣子模型的脆弱性 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07590v1) [paper-pdf](http://arxiv.org/pdf/2506.07590v1)

**Authors**: Jiacheng Shi, Yanfu Zhang, Huajie Shao, Ashley Gao

**Abstract**: Recent advancements in diffusion models have enabled high-fidelity and photorealistic image generation across diverse applications. However, these models also present security and privacy risks, including copyright violations, sensitive information leakage, and the creation of harmful or offensive content that could be exploited maliciously. In this study, we uncover a novel security threat where an attacker leverages diffusion model APIs to generate synthetic images, which are then used to train a high-performing substitute model. This enables the attacker to execute model extraction and transfer-based adversarial attacks on black-box classification models with minimal queries, without needing access to the original training data. The generated images are sufficiently high-resolution and diverse to train a substitute model whose outputs closely match those of the target model. Across the seven benchmarks, including CIFAR and ImageNet subsets, our method shows an average improvement of 27.37% over state-of-the-art methods while using just 0.01 times of the query budget, achieving a 98.68% success rate in adversarial attacks on the target model.

摘要: 扩散模型的最新进展使各种应用能够生成高保真度和真实感的图像。然而，这些模型也存在安全和隐私风险，包括版权侵犯、敏感信息泄露以及创建可能被恶意利用的有害或攻击性内容。在这项研究中，我们发现了一种新型安全威胁，攻击者利用扩散模型API来生成合成图像，然后使用合成图像来训练高性能的替代模型。这使攻击者能够以最少的查询对黑匣子分类模型执行模型提取和基于传输的对抗攻击，而无需访问原始训练数据。生成的图像具有足够高的分辨率和多样性，可以训练出一个替代模型，其输出与目标模型的输出密切匹配。在包括CIFAR和ImageNet子集在内的七个基准测试中，我们的方法比最先进的方法平均提高了27.37%，同时仅使用了0.01倍的查询预算，在目标模型上实现了98.68%的对抗攻击成功率。



## **12. MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity**

MalGEN：一个用于网络安全恶意软件建模的生成代理框架 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07586v1) [paper-pdf](http://arxiv.org/pdf/2506.07586v1)

**Authors**: Bikash Saha, Sandeep Kumar Shukla

**Abstract**: The dual use nature of Large Language Models (LLMs) presents a growing challenge in cybersecurity. While LLM enhances automation and reasoning for defenders, they also introduce new risks, particularly their potential to be misused for generating evasive, AI crafted malware. Despite this emerging threat, the research community currently lacks controlled and extensible tools that can simulate such behavior for testing and defense preparation. We present MalGEN, a multi agent framework that simulates coordinated adversarial behavior to generate diverse, activity driven malware samples. The agents work collaboratively to emulate attacker workflows, including payload planning, capability selection, and evasion strategies, within a controlled environment built for ethical and defensive research. Using MalGEN, we synthesized ten novel malware samples and evaluated them against leading antivirus and behavioral detection engines. Several samples exhibited stealthy and evasive characteristics that bypassed current defenses, validating MalGEN's ability to model sophisticated and new threats. By transforming the threat of LLM misuse into an opportunity for proactive defense, MalGEN offers a valuable framework for evaluating and strengthening cybersecurity systems. The framework addresses data scarcity, enables rigorous testing, and supports the development of resilient and future ready detection strategies.

摘要: 大型语言模型（LLM）的双重用途性质给网络安全带来了越来越大的挑战。虽然LLM增强了防御者的自动化和推理，但它们也带来了新的风险，特别是它们被滥用来生成规避的、人工智能精心设计的恶意软件的可能性。尽管存在这种新出现的威胁，但研究界目前缺乏可以模拟此类行为以进行测试和防御准备的受控和可扩展的工具。我们介绍了Malgen，这是一个多代理框架，可以模拟协调的对抗行为，以生成多样化的、活动驱动的恶意软件样本。这些代理在为道德和防御研究而构建的受控环境中协作模拟攻击者的工作流程，包括有效负载规划、能力选择和规避策略。使用Malgen，我们合成了十个新型恶意软件样本，并针对领先的防病毒和行为检测引擎对其进行了评估。几个样本表现出绕过当前防御的隐身和规避特征，验证了Malgen建模复杂和新威胁的能力。通过将LLM滥用的威胁转化为积极防御的机会，Malgen为评估和加强网络安全系统提供了一个宝贵的框架。该框架解决了数据稀缺问题，实现了严格的测试，并支持开发有弹性且面向未来的检测策略。



## **13. HSF: Defending against Jailbreak Attacks with Hidden State Filtering**

HSF：利用隐藏状态过滤防御越狱攻击 cs.CR

WWW2025 WSAI BESTPAPER

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2409.03788v2) [paper-pdf](http://arxiv.org/pdf/2409.03788v2)

**Authors**: Cheng Qian, Hainan Zhang, Lei Sha, Zhiming Zheng

**Abstract**: With the growing deployment of LLMs in daily applications like chatbots and content generation, efforts to ensure outputs align with human values and avoid harmful content have intensified. However, increasingly sophisticated jailbreak attacks threaten this alignment, aiming to induce unsafe outputs. Current defense efforts either focus on prompt rewriting or detection, which are limited in effectiveness due to the various design of jailbreak prompts, or on output control and detection, which are computationally expensive as they require LLM inference. Therefore, designing a pre-inference defense method that resists diverse jailbreak prompts is crucial for preventing LLM jailbreak attacks. We observe that jailbreak attacks, safe queries, and harmful queries exhibit different clustering patterns within the LLM's hidden state representation space. This suggests that by leveraging the LLM's hidden state representational capabilities, we can analyze the LLM's forthcoming behavior and proactively intervene for defense. In this paper, we propose a jailbreak attack defense strategy based on a Hidden State Filter (HSF), a lossless architectural defense mechanism that enables the model to preemptively identify and reject adversarial inputs before the inference process begins. We activate its defensive potential through an additional plugin module, effectively framing the defense task as a classification problem. Experimental results on two benchmark datasets, utilizing three different LLMs, show that HSF significantly enhances resilience against six cutting-edge jailbreak attacks. It significantly reduces the success rate of jailbreak attacks while minimally impacting responses to benign user queries, with negligible inference overhead, and outperforming defense baselines.Our code and data are available at https://anonymous.4open.science/r/Hidden-State-Filtering-8652/

摘要: 随着LLM在聊天机器人和内容生成等日常应用程序中的部署越来越多，确保产出与人类价值观保持一致并避免有害内容的努力得到了加强。然而，越来越复杂的越狱攻击威胁着这种一致，旨在引发不安全的输出。当前的防御工作要么集中在提示重写或检测上，由于越狱提示的各种设计，这些工作的有效性受到限制，要么集中在输出控制和检测上，因为它们需要LLM推理，计算成本很高。因此，设计一种抵御不同越狱提示的预推理防御方法对于防止LLM越狱攻击至关重要。我们观察到，越狱攻击，安全的查询，有害的查询表现出不同的聚类模式在LLM的隐藏状态表示空间。这表明，通过利用LLM的隐藏状态表示能力，我们可以分析LLM即将发生的行为，并主动干预防御。在本文中，我们提出了一种基于隐藏状态过滤器（HSF）的越狱攻击防御策略，这是一种无损的架构防御机制，使模型能够在推理过程开始之前抢先识别和拒绝敌对输入。我们通过一个额外的插件模块激活其防御潜力，有效地将防御任务视为一个分类问题。利用三种不同的LLM对两个基准数据集的实验结果表明，HSF显着增强了针对六种尖端越狱攻击的弹性。它显着降低了越狱攻击的成功率，同时对良性用户查询的响应的影响最小，推断费用可以忽略不计，并且优于防御基线。我们的代码和数据可在https://anonymous.4open.science/r/Hidden-State-Filtering-8652/上获取



## **14. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

攻击基础模型的注意力会扰乱下游任务 cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05394v2) [paper-pdf](http://arxiv.org/pdf/2506.05394v2)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention

摘要: 基础模型代表了人工智能领域最突出、最新的范式转变。基础模型是大型模型，基于广泛的数据进行训练，可以在许多下游任务中提供高准确性，通常无需微调。因此，CLIP、DINO或Vision Transfomers（ViT）等型号正在成为许多工业人工智能应用的基石。然而，对预训练的基础模型的依赖也带来了严重的安全问题，因为这些模型容易受到对抗攻击。此类攻击涉及故意设计的输入，旨在欺骗人工智能系统，危及其可靠性。本文研究了视觉基础模型的漏洞，特别关注CLIP和ViT，并探讨了对抗性攻击到下游任务的可转移性。我们引入了一种新颖的攻击，以任务不可知的方式针对基于转换器的架构的结构。我们展示了我们对几个下游任务的攻击的有效性：分类、字幕、图像/文本检索、分割和深度估计。代码可访问：https://github.com/HondamunigePrasannaSilva/attack-attention



## **15. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

通过在线自玩强化学习来追逐移动目标，以实现更安全的语言模型 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07468v1) [paper-pdf](http://arxiv.org/pdf/2506.07468v1)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).

摘要: 传统语言模型（LM）安全对齐依赖于反应性、不相交的过程：攻击者利用静态模型，然后进行防御性微调以修补暴露的漏洞。这种顺序方法造成了不匹配--攻击者过度适应过时的防御，而防御者则永远落后于新兴威胁。为了解决这个问题，我们提出了Self-RedTeam，这是一种在线自玩强化学习算法，攻击者和防御者代理通过持续的交互共同进化。我们将安全调整视为一个两人零和游戏，其中单一模型在攻击者和防御者角色之间交替--生成对抗性提示并防范它们--而奖励LM则判定结果。这实现了动态协同适应。我们以零和游戏的博弈论框架为基础，建立了一个理论安全保证，这激励了我们的方法的设计：如果自我游戏收敛于纳什均衡，防御者将可靠地对任何对抗输入产生安全反应。从经验上看，与针对静态防御者训练的攻击者相比，Self-RedTeam发现了更多样化的攻击（+21.8%SBERT），并在安全基准上实现了更高的稳健性（例如，WildJailBreak上+65.5%）比防守者训练对抗静态攻击者。我们进一步提出隐藏的思想链，允许代理人私下计划，这可以增强对抗多样性并减少过度拒绝。我们的结果促使LM安全培训从反应性修补转向主动协同进化，通过多代理强化学习（MARL）实现LM的可扩展、自主和稳健的自我改进。



## **16. A Red Teaming Roadmap Towards System-Level Safety**

迈向系统级安全的红色团队路线图 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05376v2) [paper-pdf](http://arxiv.org/pdf/2506.05376v2)

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future.

摘要: 实现请求拒绝的大型语言模型（LLM）保障措施已成为一种广泛采用的针对滥用的缓解策略。在对抗性机器学习和人工智能安全的交叉点上，红色防护有效地识别了最先进的再培训LL中的关键漏洞。然而，我们认为，许多关于LLM红色团队的会议提交的文件总体上并没有优先考虑正确的研究问题。首先，针对明确的产品安全规范进行测试应该比抽象的社会偏见或道德原则更优先。其次，红色团队应该优先考虑现实的威胁模型，这些模型代表不断扩大的风险格局以及真正的攻击者可能会做什么。最后，我们认为系统级安全是推进红色团队研究的必要步骤，因为人工智能模型呈现了新的威胁以及威胁缓解的可供性（例如，检测和禁止恶意用户）一旦置于部署上下文中。为了让红色团队研究充分解决人工智能快速发展当今和不久的将来出现的一系列新威胁，采取这些优先事项是必要的。



## **17. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2502.20650v3) [paper-pdf](http://arxiv.org/pdf/2502.20650v3)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获得。



## **18. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

不确定性和校准对可能性比隶属推理攻击的影响 cs.IT

16 pages, 28 figures

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2402.10686v5) [paper-pdf](http://arxiv.org/pdf/2402.10686v5)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在隶属推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否用于训练目标模型。在本文中，我们分析了似然比攻击（LiRA）的性能在一个信息理论框架内，允许调查的影响任意的不确定性在真实的数据生成过程中，由有限的训练数据集造成的认知不确定性，和目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型收到的信息反馈越来越少：置信向量（CV）披露，其中输出概率向量被释放;真实标签置信度（TLC）披露，其中只有分配给真实标签的概率由模型提供;以及决策集（DS）公开，其中如在共形预测中一样产生自适应预测集。我们得出了MIA对手的优势界限，旨在深入了解不确定性和校准对MIA有效性的影响。仿真结果表明，推导出的分析界能够很好地预测MIA的有效性。



## **19. Defending Against Diverse Attacks in Federated Learning Through Consensus-Based Bi-Level Optimization**

通过基于启发的双层优化防御联邦学习中的各种攻击 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2412.02535v2) [paper-pdf](http://arxiv.org/pdf/2412.02535v2)

**Authors**: Nicolás García Trillos, Aditya Kumar Akash, Sixu Li, Konstantin Riedl, Yuhua Zhu

**Abstract**: Adversarial attacks pose significant challenges in many machine learning applications, particularly in the setting of distributed training and federated learning, where malicious agents seek to corrupt the training process with the goal of jeopardizing and compromising the performance and reliability of the final models. In this paper, we address the problem of robust federated learning in the presence of such attacks by formulating the training task as a bi-level optimization problem. We conduct a theoretical analysis of the resilience of consensus-based bi-level optimization (CB$^2$O), an interacting multi-particle metaheuristic optimization method, in adversarial settings. Specifically, we provide a global convergence analysis of CB$^2$O in mean-field law in the presence of malicious agents, demonstrating the robustness of CB$^2$O against a diverse range of attacks. Thereby, we offer insights into how specific hyperparameter choices enable to mitigate adversarial effects. On the practical side, we extend CB$^2$O to the clustered federated learning setting by proposing FedCB$^2$O, a novel interacting multi-particle system, and design a practical algorithm that addresses the demands of real-world applications. Extensive experiments demonstrate the robustness of the FedCB$^2$O algorithm against label-flipping attacks in decentralized clustered federated learning scenarios, showcasing its effectiveness in practical contexts.

摘要: 对抗性攻击对许多机器学习应用程序构成了重大挑战，特别是在分布式训练和联邦学习的环境中，恶意代理试图破坏训练过程，目的是危害和损害最终模型的性能和可靠性。在本文中，我们通过将训练任务描述为双层优化问题来解决存在此类攻击时的鲁棒联邦学习问题。我们对对抗环境下基于共识的双层优化（CB $' 2$O）的弹性进行了理论分析，这是一种交互式多粒子元启发式优化方法。具体来说，我们提供了在存在恶意代理的情况下平均场定律中CB$#2$O的全球收敛分析，证明了CB$#2$O对各种攻击的稳健性。因此，我们深入了解特定的超参数选择如何减轻对抗影响。在实践方面，我们通过提出FedCB $#2$O（一种新型交互多粒子系统）将CB $#2$O扩展到集群联邦学习环境，并设计了一种可满足现实世界应用程序需求的实用算法。大量实验证明了FedCB $' 2$O算法在去中心化集群联邦学习场景中对抗标签翻转攻击的稳健性，展示了其在实际环境中的有效性。



## **20. Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation**

具有隐形语义操纵的视觉语言模型的后门攻击 cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07214v1) [paper-pdf](http://arxiv.org/pdf/2506.07214v1)

**Authors**: Zhiyuan Zhong, Zhen Sun, Yepang Liu, Xinlei He, Guanhong Tao

**Abstract**: Vision Language Models (VLMs) have shown remarkable performance, but are also vulnerable to backdoor attacks whereby the adversary can manipulate the model's outputs through hidden triggers. Prior attacks primarily rely on single-modality triggers, leaving the crucial cross-modal fusion nature of VLMs largely unexplored. Unlike prior work, we identify a novel attack surface that leverages cross-modal semantic mismatches as implicit triggers. Based on this insight, we propose BadSem (Backdoor Attack with Semantic Manipulation), a data poisoning attack that injects stealthy backdoors by deliberately misaligning image-text pairs during training. To perform the attack, we construct SIMBad, a dataset tailored for semantic manipulation involving color and object attributes. Extensive experiments across four widely used VLMs show that BadSem achieves over 98% average ASR, generalizes well to out-of-distribution datasets, and can transfer across poisoning modalities. Our detailed analysis using attention visualization shows that backdoored models focus on semantically sensitive regions under mismatched conditions while maintaining normal behavior on clean inputs. To mitigate the attack, we try two defense strategies based on system prompt and supervised fine-tuning but find that both of them fail to mitigate the semantic backdoor. Our findings highlight the urgent need to address semantic vulnerabilities in VLMs for their safer deployment.

摘要: 视觉语言模型（VLM）已表现出出色的性能，但也容易受到后门攻击，对手可以通过隐藏触发器操纵模型的输出。之前的攻击主要依赖于单模式触发，使得VLM的关键跨模式融合本质基本上没有被探索。与之前的工作不同，我们发现了一种新颖的攻击表面，它利用跨模式语义不匹配作为隐式触发器。基于这一见解，我们提出了BadSem（具有语义操纵的后门攻击），这是一种数据中毒攻击，通过在训练期间故意错位图像-文本对来注入隐形后门。为了执行攻击，我们构建了SIMBad，这是一个专为涉及颜色和对象属性的语义操作而定制的数据集。对四种广泛使用的VLM进行的广泛实验表明，BadSem的平均ASC率超过98%，很好地推广到分布外数据集，并且可以跨中毒模式传输。我们使用注意力可视化进行的详细分析表明，后门模型在不匹配的条件下专注于语义敏感区域，同时在干净的输入上保持正常行为。为了减轻攻击，我们尝试了两种基于系统提示和监督微调的防御策略，但发现这两种策略都未能减轻语义后门。我们的研究结果凸显了迫切需要解决VLM中的语义漏洞，以更安全地部署它们。



## **21. Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models**

质量多样性红色团队化：针对大型语言模型自动生成高质量且多样化的攻击者 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07121v1) [paper-pdf](http://arxiv.org/pdf/2506.07121v1)

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs.

摘要: 确保大型语言模型（LLM）的安全性非常重要。红色团队--一种识别引发目标LLM有害反应的对抗提示的系统方法--已成为一种至关重要的安全评估方法。在此框架下，对抗提示的多样性对于全面的安全评估至关重要。我们发现以前的红色团队方法可能存在两个关键限制。首先，他们经常通过词频或句子嵌入相似度等简单化指标来追求多样性，这可能无法捕捉攻击策略中有意义的变化。其次，训练单一攻击者模型的常见做法限制了潜在攻击风格和风险类别的覆盖范围。本文介绍了质量多样性红色团队（QDRT），这是一个旨在解决这些限制的新框架。QDRT通过行为条件训练实现目标驱动的多样性，并以开放式方式实现行为回放缓冲区。此外，它还培训了多个专业攻击者，能够在不同的风格和风险类别中生成高质量的攻击。我们的经验评估表明，QDRT生成的攻击更多样化，对各种目标LLM更有效，包括GPT-2，Llama-3，Gemma-2和Qwen2.5。这项工作通过提供一种系统有效的方法来自动化红队，最终支持LLM的负责任部署，从而推进了LLM安全领域。



## **22. D2R: dual regularization loss with collaborative adversarial generation for model robustness**

D2R：双正则化损失与协作对抗生成模型鲁棒性 cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07056v1) [paper-pdf](http://arxiv.org/pdf/2506.07056v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Rajiv Ranjan, Zhanxing Zhu, Vaclav Snasel, Varun Ojha

**Abstract**: The robustness of Deep Neural Network models is crucial for defending models against adversarial attacks. Recent defense methods have employed collaborative learning frameworks to enhance model robustness. Two key limitations of existing methods are (i) insufficient guidance of the target model via loss functions and (ii) non-collaborative adversarial generation. We, therefore, propose a dual regularization loss (D2R Loss) method and a collaborative adversarial generation (CAG) strategy for adversarial training. D2R loss includes two optimization steps. The adversarial distribution and clean distribution optimizations enhance the target model's robustness by leveraging the strengths of different loss functions obtained via a suitable function space exploration to focus more precisely on the target model's distribution. CAG generates adversarial samples using a gradient-based collaboration between guidance and target models. We conducted extensive experiments on three benchmark databases, including CIFAR-10, CIFAR-100, Tiny ImageNet, and two popular target models, WideResNet34-10 and PreActResNet18. Our results show that D2R loss with CAG produces highly robust models.

摘要: 深度神经网络模型的稳健性对于保护模型免受对抗攻击至关重要。最近的防御方法采用协作学习框架来增强模型的稳健性。现有方法的两个关键局限性是（i）通过损失函数对目标模型的指导不足;（ii）非协作对抗生成。因此，我们提出了一种双重正规化损失（D2 R损失）方法和一种用于对抗训练的协作对抗生成（COG）策略。D2 R损失包括两个优化步骤。对抗性分布和干净分布优化通过利用通过适当的函数空间探索获得的不同损失函数的强度来更精确地关注目标模型的分布，增强了目标模型的鲁棒性。MAG使用引导模型和目标模型之间基于梯度的协作来生成对抗样本。我们对三个基准数据库进行了广泛的实验，包括CIFAR-10、CIFAR-100、Tiny ImageNet以及两个流行的目标模型WideResNet 34 -10和PreActResNet 18。我们的结果表明，随着MAG的D2 R损失会产生高度稳健的模型。



## **23. AHSG: Adversarial Attack on High-level Semantics in Graph Neural Networks**

AHSG：对图神经网络中高级语义的对抗攻击 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2412.07468v3) [paper-pdf](http://arxiv.org/pdf/2412.07468v3)

**Authors**: Kai Yuan, Jiahao Zhang, Yidi Wang, Xiaobing Pei

**Abstract**: Adversarial attacks on Graph Neural Networks aim to perturb the performance of the learner by carefully modifying the graph topology and node attributes. Existing methods achieve attack stealthiness by constraining the modification budget and differences in graph properties. However, these methods typically disrupt task-relevant primary semantics directly, which results in low defensibility and detectability of the attack. In this paper, we propose an Adversarial Attack on High-level Semantics for Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. By combining latent representations with shared primary semantics, our model retains detectable attributes and relational patterns of the original graph while leveraging more subtle changes to carry out the attack. Then we use the Projected Gradient Descent algorithm to map the latent representations with attack effects to the adversarial graph. Through experiments on robust graph deep learning models equipped with defense strategies, we demonstrate that AHSG outperforms other state-of-the-art methods in attack effectiveness. Additionally, using Contextual Stochastic Block Models to detect the attacked graph further validates that our method preserves the primary semantics of the graph.

摘要: 对图神经网络的对抗攻击旨在通过仔细修改图布局和节点属性来扰乱学习器的性能。现有方法通过限制修改预算和图属性的差异来实现攻击隐蔽性。然而，这些方法通常直接破坏与任务相关的主要语义，从而导致攻击的防御性和可检测性较低。本文提出了一种对图神经网络高级语义的对抗攻击（AHSG），这是一种确保主要语义保留的图结构攻击模型。通过将潜在表示与共享的主要语义相结合，我们的模型保留了原始图的可检测属性和关系模式，同时利用更微妙的变化来执行攻击。然后我们使用投影梯度下降算法将具有攻击效果的潜在表示映射到对抗图。通过对配备防御策略的稳健图深度学习模型的实验，我们证明AHSG在攻击有效性方面优于其他最先进的方法。此外，使用上下文随机块模型来检测受攻击的图进一步验证了我们的方法保留了图的主要语义。



## **24. NanoZone: Scalable, Efficient, and Secure Memory Protection for Arm CCA**

NanoZone：针对Arm PCA的可扩展、高效且安全的内存保护 cs.CR

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07034v1) [paper-pdf](http://arxiv.org/pdf/2506.07034v1)

**Authors**: Shiqi Liu, Yongpeng Gao, Mingyang Zhang, Jie Wang

**Abstract**: Arm Confidential Computing Architecture (CCA) currently isolates at the granularity of an entire Confidential Virtual Machine (CVM), leaving intra-VM bugs such as Heartbleed unmitigated. The state-of-the-art narrows this to the process level, yet still cannot stop attacks that pivot within the same process, and prior intra-enclave schemes are either too slow or incompatible with CVM-style isolation. We extend CCA with a three-tier zone model that spawns an unlimited number of lightweight isolation domains inside a single process, while shielding them from kernel-space adversaries. To block domain-switch abuse, we also add a fast user-level Code-Pointer Integrity (CPI) mechanism. We developed two prototypes: a functional version on Arm's official simulator to validate resistance against intra-process and kernel-space adversaries, and a performance variant on Arm development boards evaluated for session-key isolation within server applications, in-memory key-value protection, and non-volatile-memory data isolation. NanoZone incurs roughly a 20% performance overhead while retaining 95% throughput compared to the system without fine-grained isolation.

摘要: Arm机密计算架构（CAA）目前以整个机密虚拟机（CGM）的粒度进行隔离，从而未缓解Heartbleed等虚拟机内部错误。最先进的技术将其范围缩小到流程级别，但仍然无法阻止在同一流程内进行的攻击，而且之前的飞地内方案要么太慢，要么与CGM式隔离不兼容。我们通过三层区域模型扩展了PCA，该模型在单个进程内产生无限数量的轻量级隔离域，同时保护它们免受核心空间对手的侵害。为了阻止域交换机滥用，我们还添加了快速的用户级代码指针完整性（CPI）机制。我们开发了两个原型：Arm官方模拟器上的功能版本，用于验证对进程内和核心空间对手的抵抗力，Arm开发板上的性能变体已评估服务器应用程序内的会话密钥隔离、内存内密钥值保护和非易失性内存数据隔离。与没有细粒度隔离的系统相比，NanoZone会产生大约20%的性能负担，同时保留95%的吞吐量。



## **25. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text**

对抗性重述：人工智能生成文本人性化的普遍攻击 cs.CL

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07001v1) [paper-pdf](http://arxiv.org/pdf/2506.07001v1)

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.

摘要: 大型语言模型（LLM）的能力不断增强，引发了人们对其在人工智能生成的抄袭和社会工程中滥用的担忧。虽然人们提出了各种人工智能生成的文本检测器来减轻这些风险，但许多文本检测器仍然容易受到简单规避技术（例如重述）的影响。然而，最近的检测器对此类基本攻击表现出更强的鲁棒性。在这项工作中，我们引入了对抗性重述，这是一种免训练的攻击框架，它普遍人性化任何人工智能生成的文本，以更有效地逃避检测。我们的方法利用现成的描述跟踪LLM在AI文本检测器的指导下解释AI生成的内容，生成经过专门优化以绕过检测的对抗性示例。大量实验表明，我们的攻击不仅广泛有效，而且在多个检测系统中高度可转移。例如，与简单的解释攻击相比--讽刺的是，它在RADART上将1%假阳性（T@1%F）的真阳性增加了8.57%，在Fast-DetectGPT上将15.03%--由OpenAI-RoBERTa-Large指导的对抗性解释在RADART上将T@1%F降低了64.49%，在Fast-DetectGPT上降低了98.96%。在一组不同的检测器中--包括基于神经网络、基于水印和零射击方法--在OpenAI-RoBERTa-Large的指导下，我们的攻击实现了T@1%F平均降低87.88%。我们还分析了文本质量和攻击成功之间的权衡，发现我们的方法可以显着降低检测率，但文本质量大多略有下降。鉴于日益复杂的规避技术，我们的对抗设置凸显了对更强大和更有弹性的检测策略的需求。



## **26. Boosting Adversarial Transferability via Commonality-Oriented Gradient Optimization**

通过面向公共性的梯度优化提高对抗性可移植性 cs.CV

22 pages

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.06992v1) [paper-pdf](http://arxiv.org/pdf/2506.06992v1)

**Authors**: Yanting Gao, Yepeng Liu, Junming Liu, Qi Zhang, Hongyun Zhang, Duoqian Miao, Cairong Zhao

**Abstract**: Exploring effective and transferable adversarial examples is vital for understanding the characteristics and mechanisms of Vision Transformers (ViTs). However, adversarial examples generated from surrogate models often exhibit weak transferability in black-box settings due to overfitting. Existing methods improve transferability by diversifying perturbation inputs or applying uniform gradient regularization within surrogate models, yet they have not fully leveraged the shared and unique features of surrogate models trained on the same task, leading to suboptimal transfer performance. Therefore, enhancing perturbations of common information shared by surrogate models and suppressing those tied to individual characteristics offers an effective way to improve transferability. Accordingly, we propose a commonality-oriented gradient optimization strategy (COGO) consisting of two components: Commonality Enhancement (CE) and Individuality Suppression (IS). CE perturbs the mid-to-low frequency regions, leveraging the fact that ViTs trained on the same dataset tend to rely more on mid-to-low frequency information for classification. IS employs adaptive thresholds to evaluate the correlation between backpropagated gradients and model individuality, assigning weights to gradients accordingly. Extensive experiments demonstrate that COGO significantly improves the transfer success rates of adversarial attacks, outperforming current state-of-the-art methods.

摘要: 探索有效且可转移的对抗示例对于理解视觉变形者（ViT）的特征和机制至关重要。然而，由于过度匹配，从代理模型生成的对抗性示例在黑匣子环境中通常表现出较弱的可移植性。现有的方法通过多样化扰动输入或在代理模型内应用均匀的梯度正规化来提高可移植性，但它们没有充分利用在同一任务上训练的代理模型的共享和独特特征，导致次优的传输性能。因此，增强对代理模型共享的公共信息的扰动并抑制与个体特征相关的扰动提供了提高可移植性的有效方法。因此，我们提出了一种面向共性的梯度优化策略（COGO），该策略由两个部分组成：共性增强（CE）和个性抑制（IS）。CE扰乱了中低频区域，利用了在相同数据集上训练的ViT往往更多地依赖中低频信息进行分类这一事实。IS采用自适应阈值来评估反向传播的梯度与模型个性之间的相关性，并相应地为梯度分配权重。大量实验表明，COGO显着提高了对抗性攻击的传输成功率，优于当前最先进的方法。



## **27. Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry**

重写预算：成本不对称下黑箱攻击的一般框架 cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06933v1) [paper-pdf](http://arxiv.org/pdf/2506.06933v1)

**Authors**: Mahdi Salmani, Alireza Abdollahpoorrostam, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur asymmetric costs; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. In this paper, we propose a general framework for decision-based attacks under asymmetric query costs, which we refer to as asymmetric black-box attacks. We modify two core components of existing attacks: the search strategy and the gradient estimation process. Specifically, we propose Asymmetric Search (AS), a more conservative variant of binary search that reduces reliance on high-cost queries, and Asymmetric Gradient Estimation (AGREST), which shifts the sampling distribution to favor low-cost queries. We design efficient algorithms that minimize total attack cost by balancing different query types, in contrast to earlier methods such as stealthy attacks that focus only on limiting expensive (high-cost) queries. Our method can be integrated into a range of existing black-box attacks with minimal changes. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, with improvements of up to 40% in some settings.

摘要: 对图像分类器的传统基于决策的黑匣子对抗性攻击旨在通过轻微修改输入图像同时保持较低查询数量来生成对抗性示例，其中每个查询涉及向模型发送输入并观察其输出。大多数现有方法假设所有查询的成本相同。然而，在实践中，查询可能会产生不对称的成本;例如，在内容审核系统中，某些输出类别可能会触发额外的审查、执行或处罚，使其成本比其他类别更高。虽然之前的工作已经考虑了这种不对称的成本设置，但针对这种情况的有效算法仍然不发达。本文中，我们提出了一个用于非对称查询成本下的基于决策的攻击的一般框架，我们将其称为非对称黑匣子攻击。我们修改了现有攻击的两个核心组件：搜索策略和梯度估计过程。具体来说，我们提出了不对称搜索（AS）和不对称梯度估计（CLARST），这是二分搜索的一种更保守的变体，可以减少对高成本查询的依赖，而非对称梯度估计（CLARST）则可以改变采样分布以支持低成本查询。我们设计了高效的算法，通过平衡不同的查询类型来最大限度地降低总攻击成本，而与早期的方法（例如仅专注于限制昂贵（高成本）查询的隐形攻击）形成鲜明对比。我们的方法可以只需进行最小的更改即可集成到一系列现有的黑匣子攻击中。我们对标准图像分类基准进行理论分析和实证评估。在各种成本制度中，我们的方法始终比现有方法实现更低的总查询成本和更小的扰动，在某些设置中改进高达40%。



## **28. KNN-Defense: Defense against 3D Adversarial Point Clouds using Nearest-Neighbor Search**

KNN-Defense：使用最近邻搜索防御3D对抗点云 cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06906v1) [paper-pdf](http://arxiv.org/pdf/2506.06906v1)

**Authors**: Nima Jamali, Matina Mahdizadeh Sani, Hanieh Naderi, Shohreh Kasaei

**Abstract**: Deep neural networks (DNNs) have demonstrated remarkable performance in analyzing 3D point cloud data. However, their vulnerability to adversarial attacks-such as point dropping, shifting, and adding-poses a critical challenge to the reliability of 3D vision systems. These attacks can compromise the semantic and structural integrity of point clouds, rendering many existing defense mechanisms ineffective. To address this issue, a defense strategy named KNN-Defense is proposed, grounded in the manifold assumption and nearest-neighbor search in feature space. Instead of reconstructing surface geometry or enforcing uniform point distributions, the method restores perturbed inputs by leveraging the semantic similarity of neighboring samples from the training set. KNN-Defense is lightweight and computationally efficient, enabling fast inference and making it suitable for real-time and practical applications. Empirical results on the ModelNet40 dataset demonstrated that KNN-Defense significantly improves robustness across various attack types. In particular, under point-dropping attacks-where many existing methods underperform due to the targeted removal of critical points-the proposed method achieves accuracy gains of 20.1%, 3.6%, 3.44%, and 7.74% on PointNet, PointNet++, DGCNN, and PCT, respectively. These findings suggest that KNN-Defense offers a scalable and effective solution for enhancing the adversarial resilience of 3D point cloud classifiers. (An open-source implementation of the method, including code and data, is available at https://github.com/nimajam41/3d-knn-defense).

摘要: 深度神经网络（DNN）在分析3D点云数据方面表现出出色的性能。然而，它们对对抗攻击（例如点下降、移动和添加）的脆弱性对3D视觉系统的可靠性构成了严峻的挑战。这些攻击可能会损害点云的语义和结构完整性，使许多现有的防御机制无效。为了解决这一问题，提出了一种基于多维假设和特征空间最近邻搜索的防御策略KNN-Defense。该方法不是重建表面几何形状或强制执行均匀点分布，而是通过利用训练集中邻近样本的语义相似性来恢复受干扰的输入。KNN-Defense是轻量级且计算效率高的，能够快速推理并适合实时和实际应用。Model Net 40数据集的经验结果表明，KNN-Defense显着提高了各种攻击类型的鲁棒性。特别是，在点落攻击下（许多现有方法由于有针对性地删除关键点而表现不佳），所提出的方法在PointNet、PointNet++、DGCNN和PCT上分别实现了20.1%、3.6%、3.44%和7.74%的准确性提高。这些发现表明，KNN-Defense提供了一种可扩展且有效的解决方案来增强3D点云分类器的对抗弹性。(An该方法的开源实现（包括代码和数据）可访问https：//github.com/nimajam41/3d-knn-defense）。



## **29. Robustifying Vision-Language Models via Dynamic Token Reweighting**

通过动态令牌重新加权来增强视觉语言模型 cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2505.17132v2) [paper-pdf](http://arxiv.org/pdf/2505.17132v2)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. (warning: this paper contains potentially harmful content generated by VLMs.)

摘要: 大型视觉语言模型（VLM）极易受到越狱攻击，这些攻击利用视觉与文本交互来绕过安全护栏。在本文中，我们提出了DTR，这是一种新型的推理时防御，通过优化模型的key-Value（KV）缓存来减轻多模式越狱攻击。我们不是依赖精心策划的安全特定数据或昂贵的图像到文本转换，而是引入了视觉模式引发的安全相关分布转变的新公式。该公式使DTR能够动态调整视觉令牌权重，最大限度地减少对抗视觉输入的影响，同时保留模型的一般能力和推理效率。对各种VLM和攻击基准的广泛评估表明，\sys在攻击稳健性和良性任务性能方面都优于现有防御，标志着在多模式基础模型中首次成功应用KV缓存优化来增强安全性。（警告：本文包含VLM生成的潜在有害内容。）



## **30. Can In-Context Reinforcement Learning Recover From Reward Poisoning Attacks?**

上下文强化学习能否从奖励中毒攻击中恢复？ cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06891v1) [paper-pdf](http://arxiv.org/pdf/2506.06891v1)

**Authors**: Paulius Sasnauskas, Yiğit Yalın, Goran Radanović

**Abstract**: We study the corruption-robustness of in-context reinforcement learning (ICRL), focusing on the Decision-Pretrained Transformer (DPT, Lee et al., 2023). To address the challenge of reward poisoning attacks targeting the DPT, we propose a novel adversarial training framework, called Adversarially Trained Decision-Pretrained Transformer (AT-DPT). Our method simultaneously trains an attacker to minimize the true reward of the DPT by poisoning environment rewards, and a DPT model to infer optimal actions from the poisoned data. We evaluate the effectiveness of our approach against standard bandit algorithms, including robust baselines designed to handle reward contamination. Our results show that the proposed method significantly outperforms these baselines in bandit settings, under a learned attacker. We additionally evaluate AT-DPT on an adaptive attacker, and observe similar results. Furthermore, we extend our evaluation to the MDP setting, confirming that the robustness observed in bandit scenarios generalizes to more complex environments.

摘要: 我们研究上下文强化学习（ICRL）的破坏鲁棒性，重点关注决策预训练的Transformer（DPT，Lee等人，2023年）。为了应对针对DPT的奖励中毒攻击的挑战，我们提出了一种新型的对抗训练框架，称为对抗训练决策预训练Transformer（AT-DPT）。我们的方法同时训练攻击者通过中毒环境奖励来最小化DPT的真实奖励，并训练DPT模型从中毒数据中推断最佳动作。我们评估我们的方法针对标准强盗算法的有效性，包括旨在处理奖励污染的稳健基线。我们的结果表明，在有经验的攻击者的情况下，所提出的方法在强盗环境中的表现显着优于这些基线。我们还对自适应攻击者进行了AT-DPT评估，并观察到类似的结果。此外，我们将评估扩展到MDP设置，确认在强盗场景中观察到的稳健性可以推广到更复杂的环境。



## **31. Watermark under Fire: A Robustness Evaluation of LLM Watermarking**

炮火下的水印：LLM水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2411.13425v3) [paper-pdf](http://arxiv.org/pdf/2411.13425v3)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 人们提出了各种水印方法（“水印”）来识别LLM生成的文本;然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的探讨：i）各种水印的优点/局限性是什么，尤其是它们的攻击鲁棒性？ii）各种设计选择如何影响其稳健性？iii）如何在敌对环境中最佳操作水印？为了填补这一空白，我们系统化了现有的LLM水印和水印删除攻击，绘制出它们的设计空间。然后，我们开发WaterPark，这是一个统一平台，集成了10种最先进的水印和12种代表性攻击。更重要的是，通过利用WaterPark，我们对现有的水印进行了全面评估，揭示了各种设计选择对其攻击稳健性的影响。我们进一步探索在敌对环境中操作水印的最佳实践。我们相信我们的研究揭示了当前的LLM水印技术，而WaterPark则是促进未来研究的宝贵测试平台。



## **32. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

LLM攻击者：使用大型语言模型增强自动驾驶的闭环对抗场景生成 cs.LG

Accepted as a regular paper at IEEE TITS 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.15850v2) [paper-pdf](http://arxiv.org/pdf/2501.15850v2)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.

摘要: 确保和提高自动驾驶系统（ADS）的安全性对于部署高度自动化的车辆至关重要，特别是在安全关键事件中。为了解决稀有性问题，开发了对抗性场景生成方法，其中操纵交通参与者的行为以诱导安全关键事件。然而，现有的方法仍然面临两个限制。首先，对抗参与者的识别直接影响生成的有效性。然而，现实世界场景的复杂性，众多的参与者和不同的行为，使识别具有挑战性。其次，生成的安全关键场景的潜力，以不断提高ADS的性能仍然没有得到充分的探索。为了解决这些问题，我们提出了LLM攻击者：一个利用大型语言模型（LLM）的闭环对抗场景生成框架。具体而言，多个LLM代理被设计和协调以识别最佳攻击者。然后，对攻击者的轨迹进行优化以生成对抗场景。这些场景根据ADS的性能进行迭代细化，形成反馈循环来改进ADS。实验结果表明，LLM攻击者比其他方法可以创建更危险的场景，使用它训练的ADS的碰撞率是使用正常场景训练的一半。这表明LLM攻击者有能力测试和增强ADS的安全性和稳健性。视频演示请访问：https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view。



## **33. Homophily-Driven Sanitation View for Robust Graph Contrastive Learning**

基于同质驱动的卫生观点，实现稳健的图形对比学习 cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2307.12555v2) [paper-pdf](http://arxiv.org/pdf/2307.12555v2)

**Authors**: Yulin Zhu, Xing Ai, Yevgeniy Vorobeychik, Kai Zhou

**Abstract**: We investigate adversarial robustness of unsupervised Graph Contrastive Learning (GCL) against structural attacks. First, we provide a comprehensive empirical and theoretical analysis of existing attacks, revealing how and why they downgrade the performance of GCL. Inspired by our analytic results, we present a robust GCL framework that integrates a homophily-driven sanitation view, which can be learned jointly with contrastive learning. A key challenge this poses, however, is the non-differentiable nature of the sanitation objective. To address this challenge, we propose a series of techniques to enable gradient-based end-to-end robust GCL. Moreover, we develop a fully unsupervised hyperparameter tuning method which, unlike prior approaches, does not require knowledge of node labels. We conduct extensive experiments to evaluate the performance of our proposed model, GCHS (Graph Contrastive Learning with Homophily-driven Sanitation View), against two state of the art structural attacks on GCL. Our results demonstrate that GCHS consistently outperforms all state of the art baselines in terms of the quality of generated node embeddings as well as performance on two important downstream tasks.

摘要: 我们研究了无监督图对比学习（GCL）针对结构性攻击的对抗鲁棒性。首先，我们对现有攻击进行了全面的实证和理论分析，揭示了它们如何以及为何降低GCL的性能。受我们分析结果的启发，我们提出了一个强大的GCL框架，该框架集成了同亲驱动的卫生观点，可以与对比学习联合学习。然而，这带来的一个关键挑战是卫生目标的不可区分性。为了应对这一挑战，我们提出了一系列技术来实现基于梯度的端到端稳健GCL。此外，我们开发了一种完全无监督的超参数调整方法，与之前的方法不同，该方法不需要节点标签的知识。我们进行了广泛的实验来评估我们提出的模型GCHS（具有同质性驱动的卫生视图的图形对比学习）针对对GCL的两种最新结构性攻击的性能。我们的结果表明，GCHS在生成的节点嵌入的质量以及两项重要下游任务的性能方面始终优于所有最新基线。



## **34. Refining Adaptive Zeroth-Order Optimization at Ease**

轻松细化自适应零阶优化 cs.LG

Published as a conference paper at ICML 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.01014v2) [paper-pdf](http://arxiv.org/pdf/2502.01014v2)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (a) the first analysis to the variance reduction of first moment estimate in ZO optimization, (b) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (c) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (d) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

摘要: 最近，零阶（Zero）优化在梯度信息无法访问或负担不起的场景中发挥着至关重要的作用，例如黑匣子系统和资源受限的环境。虽然现有的自适应方法（例如ZO-AdaMM）已经表现出了希望，但它们从根本上受到优化过程中矩信息利用不足的限制，通常导致收敛性能不佳。为了克服这些限制，本文引入了细化自适应零阶优化（R-AdaZR）。具体来说，我们首先展示了一次矩估计对Zero梯度估计的未开发方差降低效果，这提高了Zero更新的准确性和稳定性。然后，我们根据这些方差降低的梯度估计来细化二次矩估计，以更好地捕捉优化景观的几何形状，从而能够更有效地扩展Zero更新。我们提出了严格的理论分析，以表明（a）对Zero优化中一阶矩估计的方差缩减的第一次分析，（b）改进的二阶矩估计，其更准确地逼近其无方差理想，（c）自适应Zero方法的第一个方差感知收敛框架，这可能是独立的兴趣，以及（d）R-AdaZR比ZO-AdaMM等现有基线更快的收敛。我们的广泛实验，包括合成问题、黑匣子对抗攻击和大型语言模型（LLM）的内存高效微调，进一步验证了R-Adazo的卓越收敛性，表明R-Adazo为现实世界的Zero优化挑战提供了改进的解决方案。



## **35. On Adversarial Robustness of Language Models in Transfer Learning**

迁移学习中语言模型的对抗鲁棒性 cs.CL

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.00066v2) [paper-pdf](http://arxiv.org/pdf/2501.00066v2)

**Authors**: Bohdan Turbal, Anastasiia Mazur, Jiaxu Zhao, Mykola Pechenizkiy

**Abstract**: We investigate the adversarial robustness of LLMs in transfer learning scenarios. Through comprehensive experiments on multiple datasets (MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias) and various model architectures (BERT, RoBERTa, GPT-2, Gemma, Phi), we reveal that transfer learning, while improving standard performance metrics, often leads to increased vulnerability to adversarial attacks. Our findings demonstrate that larger models exhibit greater resilience to this phenomenon, suggesting a complex interplay between model size, architecture, and adaptation methods. Our work highlights the crucial need for considering adversarial robustness in transfer learning scenarios and provides insights into maintaining model security without compromising performance. These findings have significant implications for the development and deployment of LLMs in real-world applications where both performance and robustness are paramount.

摘要: 我们研究了迁移学习场景中LLM的对抗稳健性。通过对多个数据集（MBIB仇恨言论、MBIB政治偏见、MBIB性别偏见）和各种模型架构（BERT、RoBERTa、GPT-2、Gemma、Phi）的全面实验，我们揭示了迁移学习在提高标准性能指标的同时，往往会导致对抗性攻击的脆弱性增加。我们的研究结果表明，较大的模型对这种现象表现出更大的弹性，这表明模型大小、架构和适应方法之间存在复杂的相互作用。我们的工作强调了在迁移学习场景中考虑对抗稳健性的迫切需要，并提供了在不影响性能的情况下维护模型安全性的见解。这些发现对于在性能和稳健性都至关重要的现实应用程序中开发和部署LLM具有重大影响。



## **36. RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking**

红女王：保护大型语言模型免受隐藏的多回合越狱 cs.CR

Accepted in ACL 2025 Findings

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2409.17458v2) [paper-pdf](http://arxiv.org/pdf/2409.17458v2)

**Authors**: Yifan Jiang, Kriti Aggarwal, Tanmay Laud, Kashif Munir, Jay Pujara, Subhabrata Mukherjee

**Abstract**: The rapid progress of Large Language Models (LLMs) has opened up new opportunities across various domains and applications; yet it also presents challenges related to potential misuse. To mitigate such risks, red teaming has been employed as a proactive security measure to probe language models for harmful outputs via jailbreak attacks. However, current jailbreak attack approaches are single-turn with explicit malicious queries that do not fully capture the complexity of real-world interactions. In reality, users can engage in multi-turn interactions with LLM-based chat assistants, allowing them to conceal their true intentions in a more covert manner. To bridge this gap, we, first, propose a new jailbreak approach, RED QUEEN ATTACK. This method constructs a multi-turn scenario, concealing the malicious intent under the guise of preventing harm. We craft 40 scenarios that vary in turns and select 14 harmful categories to generate 56k multi-turn attack data points. We conduct comprehensive experiments on the RED QUEEN ATTACK with four representative LLM families of different sizes. Our experiments reveal that all LLMs are vulnerable to RED QUEEN ATTACK, reaching 87.62% attack success rate on GPT-4o and 75.4% on Llama3-70B. Further analysis reveals that larger models are more susceptible to the RED QUEEN ATTACK, with multi-turn structures and concealment strategies contributing to its success. To prioritize safety, we introduce a straightforward mitigation strategy called RED QUEEN GUARD, which aligns LLMs to effectively counter adversarial attacks. This approach reduces the attack success rate to below 1% while maintaining the model's performance across standard benchmarks. Full implementation and dataset are publicly accessible at https://github.com/kriti-hippo/red_queen.

摘要: 大型语言模型（LLM）的快速发展为各个领域和应用程序开辟了新的机会;但它也带来了与潜在滥用相关的挑战。为了减轻此类风险，红色团队已被用作一种主动安全措施，通过越狱攻击来探测语言模型的有害输出。然而，当前的越狱攻击方法是单轮的，带有显式恶意查询，无法完全捕捉现实世界交互的复杂性。事实上，用户可以与基于LLM的聊天助手进行多轮互动，使他们能够以更隐蔽的方式隐藏自己的真实意图。为了弥合这一差距，我们首先提出了一种新的越狱方法：红女王袭击。这种方法构建了一个多回合场景，以防止伤害为幌子隐藏恶意意图。我们精心设计了40个轮流变化的场景，并选择了14个有害类别来生成56，000个多回合攻击数据点。我们对《红女王袭击》进行了全面的实验，对四个不同规模的代表性LLM家族进行了实验。我们的实验表明，所有LLM都容易受到Red Queen Attack的攻击，GPT-4 o上的攻击成功率达到87.62%，Llama 3 - 70 B上的攻击成功率达到75.4%。进一步的分析表明，较大的模型更容易受到红皇后袭击的影响，多转弯结构和隐藏策略有助于其成功。为了优先考虑安全性，我们引入了一种名为RED QUEEN GUARD的简单缓解策略，该策略将LLM调整为有效对抗攻击。这种方法将攻击成功率降低至1%以下，同时在标准基准上保持模型的性能。完整的实现和数据集可在https://github.com/kriti-hippo/red_queen上公开访问。



## **37. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

有毒的：通过对抗性隐喻越狱语言模型 cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2503.00038v3) [paper-pdf](http://arxiv.org/pdf/2503.00038v3)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.

摘要: 当前的研究揭示了大型语言模型（LLM）通过越狱攻击生成有害内容的风险。然而，他们忽视了从头开始直接产生有害内容比诱导LLM将良性内容校准为有害形式更困难。在我们的研究中，我们引入了一种新颖的攻击框架，该框架利用AdVersArial meTAphoR（AVATAR）来诱导LLM校准用于越狱的恶意隐喻。具体来说，为了回答有害查询，AVATAR自适应地识别一组良性但逻辑相关的隐喻作为初始种子。然后，在这些隐喻的驱动下，目标LLM被诱导对隐喻内容进行推理和校准，从而通过直接输出有害响应或校准隐喻和专业有害内容之间的残留来越狱。实验结果表明，AVATAR可以有效且可转移的越狱LLM，并在多个高级LLM之间实现最先进的攻击成功率。



## **38. Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence**

短期对抗训练帮助法学硕士防御长期越狱攻击：理论和经验证据 cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.04204v2) [paper-pdf](http://arxiv.org/pdf/2502.04204v2)

**Authors**: Shaopeng Fu, Liang Ding, Jingfeng Zhang, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against ``long-length'' jailbreak attacks via efficient ``short-length'' AT. The code is available at https://github.com/fshp971/adv-icl.

摘要: 针对大型语言模型（LLM）的越狱攻击旨在通过精心制作的对抗性提示来诱导LLM中的有害行为。为了减轻攻击，一种方法是执行基于对抗训练（AT）的对齐，即，在一些最具对抗性的提示上训练LLM，以帮助他们学习如何在攻击下安全行事。在AT期间，对抗性提示的长度在对齐的LLM的鲁棒性中起着关键作用。虽然AT期间的长时间对抗提示可能会导致LLM强大的鲁棒性，但它们的合成非常消耗资源，这可能会限制LLM AT的应用。本文重点研究对抗性后缀越狱攻击，并揭示了为了抵御具有长度为$\Theta（M）$的对抗性后缀的越狱攻击，只需将具有长度为$\Theta（\SQRT{M}）$的对抗性后缀的提示上的LLM对齐即可。从理论上讲，我们分析了线性变换器在线性回归任务上的对抗性反向上下文学习，并证明了训练变换器的鲁棒概括界限。该界限取决于项$\Theta（\SQRT{M_{\text{Test}/M_{\text{train}}）$，其中$M_{\text{train}}$和$M_{\text{Test}$是训练和测试期间上下文中敌对干扰样本的数量。从经验上讲，我们在流行的开源LLM上进行AT，并评估它们对不同对抗后缀长度的越狱攻击的鲁棒性。结果证实了攻击成功率与越狱过程中对抗后缀长度的平方根与AT过程中的长度之比呈正相关。我们的研究结果表明，它是切实可行的，以抵御“长”越狱攻击，通过有效的“短长度”的AT。该代码可在https://github.com/fshp971/adv-icl上获取。



## **39. Stochastic Training for Side-Channel Resilient AI**

侧通道弹性人工智能的随机训练 cs.CR

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06597v1) [paper-pdf](http://arxiv.org/pdf/2506.06597v1)

**Authors**: Anuj Dubey, Aydin Aysu

**Abstract**: The confidentiality of trained AI models on edge devices is at risk from side-channel attacks exploiting power and electromagnetic emissions. This paper proposes a novel training methodology to enhance resilience against such threats by introducing randomized and interchangeable model configurations during inference. Experimental results on Google Coral Edge TPU show a reduction in side-channel leakage and a slower increase in t-scores over 20,000 traces, demonstrating robustness against adversarial observations. The defense maintains high accuracy, with about 1% degradation in most configurations, and requires no additional hardware or software changes, making it the only applicable solution for existing Edge TPUs.

摘要: 边缘设备上经过训练的人工智能模型的机密性面临着利用电力和电磁发射的侧通道攻击的风险。本文提出了一种新颖的训练方法，通过在推理期间引入随机和可互换的模型配置来增强针对此类威胁的弹性。Google Coral Edge pu上的实验结果显示，侧通道泄漏减少，t分数在20，000个轨迹上增加较慢，证明了对对抗性观察的鲁棒性。该防御保持了高准确性，在大多数配置中会降低约1%，并且不需要额外的硬件或软件更改，使其成为现有边缘pu的唯一适用解决方案。



## **40. Adapting Under Fire: Multi-Agent Reinforcement Learning for Adversarial Drift in Network Security**

在炮火下适应：网络安全中对抗漂移的多智能体强化学习 cs.CR

In Proceedings of the 22nd International Conference on Security and  Cryptography, ISBN 978-989-758-760-3, ISSN 2184-7711, pages 547-554

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06565v1) [paper-pdf](http://arxiv.org/pdf/2506.06565v1)

**Authors**: Emilia Rivas, Sabrina Saika, Ahtesham Bakht, Aritran Piplai, Nathaniel D. Bastian, Ankit Shah

**Abstract**: Evolving attacks are a critical challenge for the long-term success of Network Intrusion Detection Systems (NIDS). The rise of these changing patterns has exposed the limitations of traditional network security methods. While signature-based methods are used to detect different types of attacks, they often fail to detect unknown attacks. Moreover, the system requires frequent updates with new signatures as the attackers are constantly changing their tactics. In this paper, we design an environment where two agents improve their policies over time. The adversarial agent, referred to as the red agent, perturbs packets to evade the intrusion detection mechanism, whereas the blue agent learns new defensive policies using drift adaptation techniques to counter the attacks. Both agents adapt iteratively: the red agent responds to the evolving NIDS, while the blue agent adjusts to emerging attack patterns. By studying the model's learned policy, we offer concrete insights into drift adaptation techniques with high utility. Experiments show that the blue agent boosts model accuracy by 30% with just 2 to 3 adaptation steps using only 25 to 30 samples each.

摘要: 不断发展的攻击是网络入侵检测系统（NIDS）长期成功的关键挑战。这些不断变化的模式的兴起暴露了传统网络安全方法的局限性。虽然基于签名的方法用于检测不同类型的攻击，但它们通常无法检测未知攻击。此外，由于攻击者不断改变策略，系统需要频繁更新新签名。在本文中，我们设计了一个环境，其中两个代理会随着时间的推移改进其策略。对抗性代理（称为红色代理）扰乱数据包以逃避入侵检测机制，而蓝色代理使用漂移适应技术学习新的防御策略以对抗攻击。两个代理迭代适应：红色代理响应不断发展的NIDS，而蓝色代理则适应新出现的攻击模式。通过研究该模型的学习策略，我们为具有高实用性的漂移适应技术提供了具体见解。实验表明，蓝色代理只需2到3个适应步骤，每个步骤仅使用25到30个样本，即可将模型准确性提高30%。



## **41. Securing Traffic Sign Recognition Systems in Autonomous Vehicles**

保护自动驾驶车辆中的交通标志识别系统 cs.CV

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06563v1) [paper-pdf](http://arxiv.org/pdf/2506.06563v1)

**Authors**: Thushari Hapuarachchi, Long Dang, Kaiqi Xiong

**Abstract**: Deep Neural Networks (DNNs) are widely used for traffic sign recognition because they can automatically extract high-level features from images. These DNNs are trained on large-scale datasets obtained from unknown sources. Therefore, it is important to ensure that the models remain secure and are not compromised or poisoned during training. In this paper, we investigate the robustness of DNNs trained for traffic sign recognition. First, we perform the error-minimizing attacks on DNNs used for traffic sign recognition by adding imperceptible perturbations on training data. Then, we propose a data augmentation-based training method to mitigate the error-minimizing attacks. The proposed training method utilizes nonlinear transformations to disrupt the perturbations and improve the model robustness. We experiment with two well-known traffic sign datasets to demonstrate the severity of the attack and the effectiveness of our mitigation scheme. The error-minimizing attacks reduce the prediction accuracy of the DNNs from 99.90% to 10.6%. However, our mitigation scheme successfully restores the prediction accuracy to 96.05%. Moreover, our approach outperforms adversarial training in mitigating the error-minimizing attacks. Furthermore, we propose a detection model capable of identifying poisoned data even when the perturbations are imperceptible to human inspection. Our detection model achieves a success rate of over 99% in identifying the attack. This research highlights the need to employ advanced training methods for DNNs in traffic sign recognition systems to mitigate the effects of data poisoning attacks.

摘要: 深度神经网络（DNN）广泛用于交通标志识别，因为它们可以从图像中自动提取高级特征。这些DNN是在从未知来源获得的大规模数据集上训练的。因此，确保模型保持安全并且在训练期间不受损害或中毒非常重要。本文研究了为交通标志识别而训练的DNN的鲁棒性。首先，我们通过在训练数据上添加不可感知的扰动，对用于交通标志识别的DNN执行错误最小化攻击。然后，我们提出了一种基于数据增强的训练方法来减轻错误最小化攻击。提出的训练方法利用非线性变换来破坏扰动并提高模型的鲁棒性。我们对两个著名的交通标志数据集进行实验，以证明攻击的严重性和我们缓解方案的有效性。错误最小化攻击将DNN的预测准确率从99.90%降低到10.6%。然而，我们的缓解方案成功地将预测准确率恢复到96.05%。此外，我们的方法在减轻错误最小化攻击方面优于对抗训练。此外，我们提出了一种检测模型，即使人类检查无法察觉干扰，也能够识别有毒数据。我们的检测模型识别攻击的成功率超过99%。这项研究强调了在交通标志识别系统中采用DNN高级训练方法的必要性，以减轻数据中毒攻击的影响。



## **42. SDN-Based False Data Detection With Its Mitigation and Machine Learning Robustness for In-Vehicle Networks**

基于SDN的车载网络虚假数据检测及其缓解和机器学习鲁棒性 cs.LG

The 34th International Conference on Computer Communications and  Networks (ICCCN 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06556v1) [paper-pdf](http://arxiv.org/pdf/2506.06556v1)

**Authors**: Long Dang, Thushari Hapuarachchi, Kaiqi Xiong, Yi Li

**Abstract**: As the development of autonomous and connected vehicles advances, the complexity of modern vehicles increases, with numerous Electronic Control Units (ECUs) integrated into the system. In an in-vehicle network, these ECUs communicate with one another using an standard protocol called Controller Area Network (CAN). Securing communication among ECUs plays a vital role in maintaining the safety and security of the vehicle. This paper proposes a robust SDN-based False Data Detection and Mitigation System (FDDMS) for in-vehicle networks. Leveraging the unique capabilities of Software-Defined Networking (SDN), FDDMS is designed to monitor and detect false data injection attacks in real-time. Specifically, we focus on brake-related ECUs within an SDN-enabled in-vehicle network. First, we decode raw CAN data to create an attack model that illustrates how false data can be injected into the system. Then, FDDMS, incorporating a Long Short Term Memory (LSTM)-based detection model, is used to identify false data injection attacks. We further propose an effective variant of DeepFool attack to evaluate the model's robustness. To countermeasure the impacts of four adversarial attacks including Fast gradient descent method, Basic iterative method, DeepFool, and the DeepFool variant, we further enhance a re-training technique method with a threshold based selection strategy. Finally, a mitigation scheme is implemented to redirect attack traffic by dynamically updating flow rules through SDN. Our experimental results show that the proposed FDDMS is robust against adversarial attacks and effectively detects and mitigates false data injection attacks in real-time.

摘要: 随着自动驾驶和互联车辆的发展，现代车辆的复杂性不断增加，系统中集成了大量电子控制单元（EC）。在车载网络中，这些EC使用称为控制器区域网络（CAN）的标准协议相互通信。确保EC之间的通信对于维护车辆的安全性发挥着至关重要的作用。本文提出了一种用于车载网络的稳健的基于SEN的错误数据检测和缓解系统（FDDMS）。FDDMS利用软件定义网络（dn）的独特功能，旨在实时监控和检测虚假数据注入攻击。具体来说，我们重点关注支持SDP的车载网络中与制动相关的MCU。首先，我们解码原始CAN数据以创建一个攻击模型，该模型说明如何将虚假数据注入系统。然后，FDDMS结合了基于长短期记忆（LSTM）的检测模型，用于识别错误数据注入攻击。我们进一步提出DeepFool攻击的有效变体来评估模型的稳健性。为了应对四种对抗攻击的影响，包括快速梯度下降法、基本迭代法、DeepFool和DeepFool变体，我们进一步增强了基于阈值的选择策略的重新训练技术方法。最后，实现了一种缓解方案，通过SDK动态更新流规则来重定向攻击流量。我们的实验结果表明，所提出的FDDMS对对抗攻击具有鲁棒性，并有效地实时检测和缓解虚假数据注入攻击。



## **43. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

越狱镜头：针对大型语言模型的越狱攻击的视觉分析 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2404.08793v2) [paper-pdf](http://arxiv.org/pdf/2404.08793v2)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Haoyu Tian, Wei Zhang, Minfeng Zhu, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.

摘要: 大型语言模型（LLM）的激增凸显了对其安全漏洞的担忧，特别是针对越狱攻击，对手设计越狱提示规避潜在滥用的安全机制。解决这些问题需要对越狱提示进行全面分析，以评估LLM的防御能力并识别潜在的弱点。然而，评估越狱表现和了解提示特征的复杂性使得这项分析变得费力。我们与领域专家合作来描述问题并提出LLM辅助框架来简化分析过程。它提供自动越狱评估，以促进性能评估并支持对提示中的组件和关键词的分析。基于该框架，我们设计了JailbreakLens，这是一个视觉分析系统，使用户能够根据目标模型探索越狱表现，对提示特征进行多层次分析，并细化提示实例以验证结果。通过案例研究、技术评估和专家访谈，我们展示了我们的系统在帮助用户评估模型安全性和识别模型弱点方面的有效性。



## **44. Membership Inference Attacks for Unseen Classes**

对隐形班级的会员推断攻击 cs.LG

Preprint

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06488v1) [paper-pdf](http://arxiv.org/pdf/2506.06488v1)

**Authors**: Pratiksha Thaker, Neil Kale, Zhiwei Steven Wu, Virginia Smith

**Abstract**: Shadow model attacks are the state-of-the-art approach for membership inference attacks on machine learning models. However, these attacks typically assume an adversary has access to a background (nonmember) data distribution that matches the distribution the target model was trained on. We initiate a study of membership inference attacks where the adversary or auditor cannot access an entire subclass from the distribution -- a more extreme but realistic version of distribution shift than has been studied previously. In this setting, we first show that the performance of shadow model attacks degrades catastrophically, and then demonstrate the promise of another approach, quantile regression, that does not have the same limitations. We show that quantile regression attacks consistently outperform shadow model attacks in the class dropout setting -- for example, quantile regression attacks achieve up to 11$\times$ the TPR of shadow models on the unseen class on CIFAR-100, and achieve nontrivial TPR on ImageNet even with 90% of training classes removed. We also provide a theoretical model that illustrates the potential and limitations of this approach.

摘要: 影子模型攻击是对机器学习模型进行隶属度推理攻击的最先进方法。然而，这些攻击通常假设对手可以访问与目标模型训练的分布相匹配的背景（非成员）数据分布。我们启动了一项成员资格推断攻击的研究，其中对手或审计者无法访问来自分布的整个子集--这是一个比之前研究的更极端但现实的分布转移版本。在这种情况下，我们首先表明影子模型攻击的性能会发生灾难性下降，然后证明另一种方法（分位数回归）的前景，该方法没有相同的限制。我们表明，分位数回归攻击在类退出设置中始终优于影子模型攻击-例如，分位数回归攻击在CIFAR-100上未见类上实现了高达11 $\times的影子模型的TPA，并且在ImageNet上实现了重要的TPA，即使删除了90%的训练类。我们还提供了一个理论模型，说明了这种方法的潜力和局限性。



## **45. Adaptive and Robust Watermark for Generative Tabular Data**

生成表格数据的自适应鲁棒水印 cs.CR

15 pages of main body, 5 figures, 5 tables

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2409.14700v2) [paper-pdf](http://arxiv.org/pdf/2409.14700v2)

**Authors**: Dung Daniel Ngo, Daniel Scott, Saheed Obitayo, Archan Ray, Akshay Seshadri, Niraj Kumar, Vamsi K. Potluru, Marco Pistoia, Manuela Veloso

**Abstract**: Recent development in generative models has demonstrated its ability to create high-quality synthetic data. However, the pervasiveness of synthetic content online also brings forth growing concerns that it can be used for malicious purpose. To ensure the authenticity of the data, watermarking techniques have recently emerged as a promising solution due to their strong statistical guarantees. In this paper, we propose a flexible and robust watermarking mechanism for generative tabular data. Specifically, a data provider with knowledge of the downstream tasks can partition the feature space into pairs of (key, value) columns. Within each pair, the data provider first uses elements in the key column to generate a randomized set of ``green'' intervals, then encourages elements of the value column to be in one of these ``green'' intervals. We show theoretically and empirically that the watermarked datasets (i) have negligible impact on the data quality and downstream utility, (ii) can be efficiently detected, (iii) are robust against multiple attacks commonly observed in data science, and (iv) maintain strong security against adversary attempting to learn the underlying watermark scheme.

摘要: 生成模型的最新发展证明了其创建高质量合成数据的能力。然而，在线合成内容的普遍性也引发了人们日益担心其可能被用于恶意目的。为了确保数据的真实性，水印技术最近因其强大的统计保证而成为一种有前途的解决方案。本文针对生成式表格数据提出了一种灵活且鲁棒的水印机制。具体来说，了解下游任务的数据提供者可以将特征空间划分为成对的（键、值）列。在每对中，数据提供者首先使用键列中的元素来生成随机的一组“绿色”间隔，然后鼓励值列的元素处于这些“绿色”间隔之一中。我们从理论上和经验上表明，带水印的数据集（i）对数据质量和下游效用的影响可以忽略不计，（ii）可以被有效检测，（iii）对数据科学中常见的多种攻击具有鲁棒性，（iv）对试图学习底层水印方案的对手保持强大的安全性。



## **46. Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets**

Fréchet放射距离（FRD）：用于比较医学成像数据集的通用指标 cs.CV

Codebase for FRD computation:  https://github.com/RichardObi/frd-score. Codebase for medical image  similarity metric evaluation framework:  https://github.com/mazurowski-lab/medical-image-similarity-metrics

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2412.01496v2) [paper-pdf](http://arxiv.org/pdf/2412.01496v2)

**Authors**: Nicholas Konz, Richard Osuala, Preeti Verma, Yuwen Chen, Hanxue Gu, Haoyu Dong, Yaqian Chen, Andrew Marshall, Lidia Garrucho, Kaisar Kushibar, Daniel M. Lang, Gene S. Kim, Lars J. Grimm, John M. Lewin, James S. Duncan, Julia A. Schnabel, Oliver Diaz, Karim Lekadir, Maciej A. Mazurowski

**Abstract**: Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fr\'echet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fr\'echet Radiomic Distance), which utilizes standardized, clinically meaningful, and interpretable image features. We show that FRD is superior to other image distribution metrics for a range of medical imaging applications, including out-of-domain (OOD) detection, the evaluation of image-to-image translation (by correlating more with downstream task performance as well as anatomical consistency and realism), and the evaluation of unconditional image generation. Moreover, FRD offers additional benefits such as stability and computational efficiency at low sample sizes, sensitivity to image corruptions and adversarial attacks, feature interpretability, and correlation with radiologist-perceived image quality. Additionally, we address key gaps in the literature by presenting an extensive framework for the multifaceted evaluation of image similarity metrics in medical imaging -- including the first large-scale comparative study of generative models for medical image translation -- and release an accessible codebase to facilitate future research. Our results are supported by thorough experiments spanning a variety of datasets, modalities, and downstream tasks, highlighting the broad potential of FRD for medical image analysis.

摘要: 确定两组图像是否属于相同或不同的分布或领域是现代医学图像分析和深度学习的一项关键任务;例如，评估图像生成模型的输出质量。目前，用于该任务的指标要么依赖于对某些下游任务（例如分段）的（潜在有偏见的）选择，要么采用与任务无关的感知指标（例如，Fr ' echet Incement Distance/DID）来自自然成像，我们表明该成像不足以捕捉解剖特征。为此，我们引入了一种为医学图像量身定制的新感知指标FRD（Frechet Radiomic Distance），它利用标准化、具有临床意义且可解释的图像特征。我们表明，对于一系列医学成像应用，FRD优于其他图像分布指标，包括域外（OOD）检测、图像到图像转换的评估（通过与下游任务性能以及解剖一致性和真实性进行更多相关），以及无条件图像生成的评估。此外，FRD还提供了额外的好处，例如低样本量下的稳定性和计算效率、对图像损坏和对抗攻击的敏感性、特征可解释性以及与放射科医生感知的图像质量的相关性。此外，我们还通过提供一个广泛的框架来解决文献中的关键空白，用于医学成像中的图像相似性指标的多方面评估（包括医学图像翻译生成模型的第一个大规模比较研究），并发布一个可访问的代码库以促进未来的研究。我们的结果得到了跨越各种数据集、模式和下游任务的彻底实验的支持，凸显了FRD在医学图像分析方面的广泛潜力。



## **47. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

ByzSecAgg：一种基于编码计算和载体承诺的联邦学习抗拜占庭安全聚合方案 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2302.09913v4) [paper-pdf](http://arxiv.org/pdf/2302.09913v4)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose ByzSecAgg, an efficient secure aggregation scheme for federated learning that is resistant to Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving the privacy of the data against colluding nodes, requires some sort of secure secret sharing. However, the communication load for secret sharing of long vectors of updates can be very high. In federated settings, where users are often edge devices with potential bandwidth constraints, excessive communication overhead is undesirable. ByzSecAgg solves this problem by partitioning local updates into smaller sub-vectors and sharing them using ramp secret sharing. However, this sharing method does not admit bilinear computations, such as pairwise distances calculations, which are needed for distance-based outlier-detection algorithms, and effective methods for mitigating Byzantine attacks. To overcome this issue, each user runs another round of ramp sharing, with a different embedding of the data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, ByzSecAgg also uses a vector commitment method, in which the commitment size remains constant (i.e., does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process. In terms of communication load, ByzSecAgg significantly outperforms the related baseline scheme, known as BREA.

摘要: 在本文中，我们提出了ByzSecAgg，这是一种用于联邦学习的高效安全聚合方案，可以抵抗拜占庭攻击和隐私泄露。处理个人更新以管理对抗行为，同时保护数据隐私以防止勾结节点，需要某种安全的秘密共享。然而，长更新载体的秘密共享的通信负载可能非常高。在联合设置中，用户通常是具有潜在带宽限制的边缘设备，因此不希望过度的通信负担。ByzSecAgg通过将本地更新划分为更小的子载体并使用斜坡秘密共享来共享它们来解决这个问题。然而，这种共享方法不允许双线性计算，例如基于距离的离群值检测算法所需的成对距离计算，以及缓解拜占庭攻击的有效方法。为了克服这个问题，每个用户运行另一轮坡道共享，并在共享多元中嵌入不同的数据。这项技术的灵感来自编码计算，可以安全计算成对距离。此外，为了维护本地更新的完整性和隐私，ByzSecAgg还使用了一种载体承诺方法，其中承诺大小保持不变（即，不会随着本地更新的长度而增加），同时允许验证秘密共享过程。在通信负载方面，ByzSecAgg显着优于相关基线方案（即BRA）。



## **48. SATversary: Adversarial Attacks on Satellite Fingerprinting**

卫星指纹：对卫星指纹的对抗性攻击 cs.CR

19 pages, 18 figures, 2 tables

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06119v1) [paper-pdf](http://arxiv.org/pdf/2506.06119v1)

**Authors**: Joshua Smailes, Sebastian Köhler, Simon Birnbach, Martin Strohmeier, Ivan Martinovic

**Abstract**: As satellite systems become increasingly vulnerable to physical layer attacks via SDRs, novel countermeasures are being developed to protect critical systems, particularly those lacking cryptographic protection, or those which cannot be upgraded to support modern cryptography. Among these is transmitter fingerprinting, which provides mechanisms by which communication can be authenticated by looking at characteristics of the transmitter, expressed as impairments on the signal.   Previous works show that fingerprinting can be used to classify satellite transmitters, or authenticate them against SDR-equipped attackers under simple replay scenarios. In this paper we build upon this by looking at attacks directly targeting the fingerprinting system, with an attacker optimizing for maximum impact in jamming, spoofing, and dataset poisoning attacks, and demonstrate these attacks on the SatIQ system designed to authenticate Iridium transmitters. We show that an optimized jamming signal can cause a 50% error rate with attacker-to-victim ratios as low as -30dB (far less power than traditional jamming) and demonstrate successful identity forgery during spoofing attacks, with an attacker successfully removing their own transmitter's fingerprint from messages. We also present a data poisoning attack, enabling persistent message spoofing by altering the data used to authenticate incoming messages to include the fingerprint of the attacker's transmitter.   Finally, we show that our model trained to optimize spoofing attacks can also be used to detect spoofing and replay attacks, even when it has never seen the attacker's transmitter before. Furthermore, this technique works even when the training dataset includes only a single transmitter, enabling fingerprinting to be used to protect small constellations and even individual satellites, providing additional protection where it is needed the most.

摘要: 随着卫星系统越来越容易受到通过SDR的物理层攻击，人们正在开发新型对策来保护关键系统，特别是那些缺乏加密保护的系统，或那些无法升级以支持现代加密的系统。其中之一是发射器指纹识别，它提供了通过查看发射器的特征（表示为信号的损害）来验证通信的机制。   之前的工作表明，指纹识别可用于对卫星发射机进行分类，或在简单的回放场景下针对配备了SDP的攻击者进行验证。在本文中，我们通过研究直接针对指纹识别系统的攻击来建立此基础，攻击者对干扰、欺骗和数据集中毒攻击的最大影响进行优化，并在旨在认证铱星发射机的SatIQ系统上演示了这些攻击。我们表明，优化的干扰信号可以导致50%的错误率，攻击者与受害者的比率低至-30分贝（功率远低于传统干扰），并在欺骗攻击期间证明了成功的身份伪造，攻击者成功地从消息中删除了自己发射机的指纹。我们还提出了一种数据中毒攻击，通过更改用于验证输入消息的数据以包括攻击者发射机的指纹来实现持续消息欺骗。   最后，我们表明，我们为优化欺骗攻击而训练的模型也可用于检测欺骗和重播攻击，即使它以前从未见过攻击者的发射机。此外，即使训练数据集仅包括单个发射机，这种技术也有效，使指纹识别能够用于保护小星座甚至单个卫星，在最需要的地方提供额外保护。



## **49. The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text**

金丝雀的回声：审计LLM生成的合成文本的隐私风险 cs.CL

42nd International Conference on Machine Learning (ICML 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2502.14921v2) [paper-pdf](http://arxiv.org/pdf/2502.14921v2)

**Authors**: Matthieu Meeus, Lukas Wutschitz, Santiago Zanella-Béguelin, Shruti Tople, Reza Shokri

**Abstract**: How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.

摘要: 大型语言模型（LLM）生成的合成数据会泄露多少有关训练样本的信息？忽视合成数据生成管道中信息流的微妙之处可能会导致错误的隐私感。在本文中，我们假设对手可以访问LLM生成的一些合成数据。我们设计了成员资格推理攻击（MIA），针对用于微调LLM的训练数据，然后用于合成数据。我们的MIA的显着性能表明合成数据泄露了有关训练数据的信息。此外，我们发现，当仅发布合成数据时，为基于模型的MIA制作的金丝雀对于隐私审计来说并不是最佳的。当提示生成有用的、分布内的合成数据时，这种不分布的金丝雀对模型的输出影响有限，从而大大降低了其有效性。为了解决这个问题，我们利用自回归模型的机制来设计具有内分布后缀和高困惑性后缀的金丝雀，从而在合成数据中留下可检测的痕迹。这增强了基于数据的MIA的能力，并对发布LLM生成的合成数据的隐私风险提供了更好的评估。



## **50. One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy**

一石二鸟：通过分散的视角增强对抗性防御 cs.LG

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2503.02169v2) [paper-pdf](http://arxiv.org/pdf/2503.02169v2)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we explore the strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of useful information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-discrepancy-based Adversarial Defense (DAD). In the training phase, DAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, which is a stone that kills two birds. MMD-OPT first serves as a guiding signal to minimize the distributional discrepancy between CEs and AEs to train a denoiser. Then, it serves as a discriminator to differentiate CEs and AEs during inference. Overall, in the inference stage, DAD consists of a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DAD outperforms current state-of-the-art (SOTA) defense methods by simultaneously improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks. Codes are publicly available at: https://github.com/tmlr-group/DAD.

摘要: 统计对抗数据检测（SADD）通过测量干净示例（CE）和AE之间的分布差异来检测即将到来的批次是否包含对抗示例（AE）。在本文中，我们通过理论上证明最小化分布差异可以帮助减少AE的预期损失来探索基于SADD的方法的优势。尽管有这些优势，但基于SADD的方法也有一个潜在的局限性：它们会丢弃被检测为AE的输入，从而导致这些输入中有用信息的丢失。为了解决这一局限性，我们提出了一种双管齐下的对抗防御方法，称为基于分布差异的对抗防御（DAD）。在训练阶段，DAD首先优化最大均值差异（MMD）的测试功效，以推导出MMD-OPT，这是一种杀死两只鸟的石头。MMD-OPT首先充当引导信号，以最小化CE和AE之间的分布差异，以训练降噪器。然后，它在推理过程中充当鉴别器来区分CE和AE。总体而言，在推理阶段，DAD由一个双管齐下的过程组成：（1）将检测到的CE直接输入分类器，（2）通过基于分布差异的降噪器从检测到的AE中去除噪音。大量实验表明，DAD通过同时提高CIFAR-10和ImageNet-1 K针对自适应白盒攻击的清晰和稳健的准确性，优于当前最先进的（SOTA）防御方法。代码可在https://github.com/tmlr-group/DAD上公开获取。



