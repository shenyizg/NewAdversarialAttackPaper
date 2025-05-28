# Latest Adversarial Attack Papers
**update at 2025-05-28 14:53:55**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AdInject: Real-World Black-Box Attacks on Web Agents via Advertising Delivery**

AdInjects：通过广告交付对Web代理进行现实世界的黑匣子攻击 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21499v1) [paper-pdf](http://arxiv.org/pdf/2505.21499v1)

**Authors**: Haowei Wang, Junjie Wang, Xiaojun Jia, Rupeng Zhang, Mingyang Li, Zhe Liu, Yang Liu, Qing Wang

**Abstract**: Vision-Language Model (VLM) based Web Agents represent a significant step towards automating complex tasks by simulating human-like interaction with websites. However, their deployment in uncontrolled web environments introduces significant security vulnerabilities. Existing research on adversarial environmental injection attacks often relies on unrealistic assumptions, such as direct HTML manipulation, knowledge of user intent, or access to agent model parameters, limiting their practical applicability. In this paper, we propose AdInject, a novel and real-world black-box attack method that leverages the internet advertising delivery to inject malicious content into the Web Agent's environment. AdInject operates under a significantly more realistic threat model than prior work, assuming a black-box agent, static malicious content constraints, and no specific knowledge of user intent. AdInject includes strategies for designing malicious ad content aimed at misleading agents into clicking, and a VLM-based ad content optimization technique that infers potential user intents from the target website's context and integrates these intents into the ad content to make it appear more relevant or critical to the agent's task, thus enhancing attack effectiveness. Experimental evaluations demonstrate the effectiveness of AdInject, attack success rates exceeding 60% in most scenarios and approaching 100% in certain cases. This strongly demonstrates that prevalent advertising delivery constitutes a potent and real-world vector for environment injection attacks against Web Agents. This work highlights a critical vulnerability in Web Agent security arising from real-world environment manipulation channels, underscoring the urgent need for developing robust defense mechanisms against such threats. Our code is available at https://github.com/NicerWang/AdInject.

摘要: 基于视觉语言模型（VLM）的Web代理通过模拟与网站的类人交互来实现复杂任务自动化迈出了重要一步。然而，它们在不受控制的Web环境中的部署会带来严重的安全漏洞。现有的针对对抗性环境注入攻击的研究通常依赖于不切实际的假设，例如直接的HTML操作、用户意图的知识或对代理模型参数的访问，限制了它们的实际适用性。在本文中，我们提出了AdInib，这是一种新颖的现实世界黑匣子攻击方法，它利用互联网广告交付将恶意内容注入到Web代理的环境中。与之前的工作相比，AdInib在一个明显更现实的威胁模型下运行，假设黑匣子代理、静态恶意内容约束并且没有对用户意图的具体了解。AdInjects包括设计恶意广告内容的策略，旨在误导代理点击，以及一种基于LMA的广告内容优化技术，该技术从目标网站的上下文推断潜在的用户意图，并将这些意图集成到广告内容中，使其看起来与代理的任务更相关或至关重要，从而增强攻击有效性。实验评估证明了AdInib的有效性，在大多数情况下攻击成功率超过60%，在某些情况下攻击成功率接近100%。这有力地表明，流行的广告交付构成了针对Web代理的环境注入攻击的强大现实载体。这项工作强调了现实世界环境操纵渠道引起的Web Agent安全中的一个关键漏洞，强调了开发针对此类威胁的强大防御机制的迫切需要。我们的代码可在https://github.com/NicerWang/AdInject上获取。



## **2. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

通过特征最佳对齐对闭源MLLM的对抗攻击 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.

摘要: 多模式大型语言模型（MLLM）仍然容易受到可转移的对抗示例的影响。虽然现有方法通常通过在对抗样本和目标样本之间对齐全局特征（例如CLIP的[LIS]标记）来实现有针对性的攻击，但它们经常忽视补丁令牌中编码的丰富本地信息。这导致次优的对齐和有限的可移植性，特别是对于闭源模型。为了解决这一局限性，我们提出了一种基于特征最优对齐的有针对性的可转移对抗攻击方法，称为FOA-Attack，以提高对抗转移能力。具体来说，在全球层面，我们引入了基于cos相似性的全球特征损失，以将对抗样本的粗粒度特征与目标样本的粗粒度特征对齐。在局部层面，鉴于变形金刚中丰富的局部表示，我们利用集群技术来提取紧凑的局部模式，以减轻冗余的局部特征。然后，我们将对抗样本和目标样本之间的局部特征对齐公式化为最优传输（OT）问题，并提出局部集群最优传输损失来细化细粒度特征对齐。此外，我们还提出了一种动态集成模型加权策略，以自适应地平衡对抗性示例生成过程中多个模型的影响，从而进一步提高可移植性。跨各种模型的广泛实验证明了所提出方法的优越性，优于最先进的方法，特别是在转移到闭源MLLM方面。该代码发布于https://github.com/jiaxiaojunQAQ/FOA-Attack。



## **3. When Are Concepts Erased From Diffusion Models?**

概念何时从扩散模型中删除？ cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.17013v3) [paper-pdf](http://arxiv.org/pdf/2505.17013v3)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.

摘要: 概念擦除，即选择性地阻止模型生成特定概念的能力，引起了越来越多的兴趣，各种方法出现了来应对这一挑战。然而，目前尚不清楚这些方法如何彻底消除目标概念。我们首先提出了扩散模型中擦除机制的两个概念模型：（i）降低生成目标概念的可能性，（ii）干扰模型的内部引导机制。为了彻底评估某个概念是否已真正从模型中删除，我们引入了一套独立评估。我们的评估框架包括对抗性攻击、新颖的探测技术以及对模型替代世代的分析，以取代被删除的概念。我们的结果揭示了最大限度地减少副作用和保持对抗提示的鲁棒性之间的紧张关系。从广义上讲，我们的工作强调了对扩散模型中擦除进行全面评估的重要性。



## **4. On the Robustness of Adversarial Training Against Uncertainty Attacks**

论对抗训练对不确定性攻击的鲁棒性 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2410.21952v2) [paper-pdf](http://arxiv.org/pdf/2410.21952v2)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.

摘要: 在学习问题中，手头任务固有的噪音阻碍了在没有一定不确定性的情况下进行推断的可能性。量化这种不确定性，无论其是否广泛使用，都假设与安全敏感应用程序高度相关。在这些场景中，保证良好（即，值得信赖的）不确定性措施，下游模块可以安全地使用这些措施来驱动最终的决策过程。然而，攻击者可能有兴趣强迫系统产生（i）高度不确定的输出，危及系统的可用性，或（ii）低不确定性估计，使系统接受不确定的样本，而这些样本将需要仔细检查（例如，人类干预）。因此，了解如何针对此类攻击获得稳健的不确定性估计变得至关重要。在这项工作中，我们从经验和理论上揭示了防御对抗性例子，即，经过精心扰动的样本会导致错误分类，还可以在常见攻击场景下保证更安全、更值得信赖的不确定性估计，而无需临时防御策略。为了支持我们的主张，我们在CIFAR-10和ImageNet数据集上评估了来自公开基准RobustBench的多个对抗稳健模型。



## **5. Attribute-Efficient PAC Learning of Sparse Halfspaces with Constant Malicious Noise Rate**

稀疏半空间中恒定恶意噪声率下的属性有效PAC学习 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21430v1) [paper-pdf](http://arxiv.org/pdf/2505.21430v1)

**Authors**: Shiwei Zeng, Jie Shen

**Abstract**: Attribute-efficient learning of sparse halfspaces has been a fundamental problem in machine learning theory. In recent years, machine learning algorithms are faced with prevalent data corruptions or even adversarial attacks. It is of central interest to design efficient algorithms that are robust to noise corruptions. In this paper, we consider that there exists a constant amount of malicious noise in the data and the goal is to learn an underlying $s$-sparse halfspace $w^* \in \mathbb{R}^d$ with $\text{poly}(s,\log d)$ samples. Specifically, we follow a recent line of works and assume that the underlying distribution satisfies a certain concentration condition and a margin condition at the same time. Under such conditions, we show that attribute-efficiency can be achieved by simple variants to existing hinge loss minimization programs. Our key contribution includes: 1) an attribute-efficient PAC learning algorithm that works under constant malicious noise rate; 2) a new gradient analysis that carefully handles the sparsity constraint in hinge loss minimization.

摘要: 稀疏半空间的属性高效学习一直是机器学习理论中的一个基本问题。近年来，机器学习算法面临着普遍的数据损坏甚至对抗性攻击。设计对噪音破坏具有鲁棒性的高效算法是最重要的。在本文中，我们认为数据中存在恒定数量的恶意噪音，目标是学习底层的$s$-稀疏半空间$w #\in \mathbb{R}' d$，带有$\text{poly}（s，\log d）$ samples。具体来说，我们遵循最近的工作路线，并假设基本分布同时满足一定的集中度条件和边际条件。在这种条件下，我们表明属性效率可以通过现有铰链损失最小化程序的简单变体来实现。我们的主要贡献包括：1）一种在恒定恶意噪音率下工作的属性高效PAC学习算法; 2）一种新的梯度分析，仔细处理铰链损失最小化中的稀疏性约束。



## **6. A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment**

决策支持系统部署前的对抗性分析框架 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21414v1) [paper-pdf](http://arxiv.org/pdf/2505.21414v1)

**Authors**: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman

**Abstract**: This paper introduces a comprehensive framework designed to analyze and secure decision-support systems trained with Deep Reinforcement Learning (DRL), prior to deployment, by providing insights into learned behavior patterns and vulnerabilities discovered through simulation. The introduced framework aids in the development of precisely timed and targeted observation perturbations, enabling researchers to assess adversarial attack outcomes within a strategic decision-making context. We validate our framework, visualize agent behavior, and evaluate adversarial outcomes within the context of a custom-built strategic game, CyberStrike. Utilizing the proposed framework, we introduce a method for systematically discovering and ranking the impact of attacks on various observation indices and time-steps, and we conduct experiments to evaluate the transferability of adversarial attacks across agent architectures and DRL training algorithms. The findings underscore the critical need for robust adversarial defense mechanisms to protect decision-making policies in high-stakes environments.

摘要: 本文介绍了一个全面的框架，旨在在部署之前分析和保护经过深度强化学习（DRL）训练的决策支持系统，通过提供对通过模拟发现的行为模式和漏洞的见解。引入的框架有助于开发精确定时和有针对性的观察扰动，使研究人员能够在战略决策背景下评估对抗性攻击的结果。我们在定制的战略游戏CyberStrike的背景下验证我们的框架、可视化代理行为并评估对抗结果。利用提出的框架，我们引入了一种系统性地发现和排名攻击对各种观察指标和时间步的影响的方法，并进行实验来评估对抗性攻击跨代理架构和DRL训练算法的可转移性。研究结果强调，迫切需要强大的对抗性防御机制来保护高风险环境中的决策政策。



## **7. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

混合专家系统鲁棒性和准确性的双模型优化方法 cs.LG

15 pages, 7 figures, accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.06832v3) [paper-pdf](http://arxiv.org/pdf/2502.06832v3)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods. The code is publicly available at https://github.com/TIML-Group/Robust-MoE-Dual-Model.

摘要: 混合专家（MoE）在利用专业专家网络执行复杂的机器学习任务方面取得了显着的成功。然而，它们对对抗攻击的敏感性给在强大应用程序中的部署带来了严峻的挑战。本文解决了如何将稳健性融入MoE同时保持高自然准确性的关键问题。我们首先分析MoE组件的漏洞，发现专家网络明显比路由器更容易受到对抗攻击。基于这一见解，我们提出了一种有针对性的鲁棒训练技术，该技术集成了一种新颖的损失函数来增强MoE的对抗鲁棒性，只需要一名额外专家的鲁棒性，而不会损害训练或推理效率。在此基础上，我们引入了一种双模型策略，该策略使用平滑参数将标准MoE模型与我们的鲁棒化MoE模型线性结合。这种方法允许灵活控制鲁棒性-准确性权衡。我们通过推导单MoE和双模型的经认证的鲁棒性界限来进一步提供理论基础。为了突破稳健性和准确性的界限，我们提出了一种新颖的双模型联合训练策略JTDMoE。这种联合训练增强了鲁棒性和准确性，超出了单独模型所能实现的能力。使用ResNet 18和Vision Transformer（ViT）架构在CIFAR-10和TinyImageNet数据集上的实验结果证明了我们提出的方法的有效性。该代码可在https://github.com/TIML-Group/Robust-MoE-Dual-Model上公开获取。



## **8. Boosting Adversarial Transferability via High-Frequency Augmentation and Hierarchical-Gradient Fusion**

通过高频增强和分层梯度融合提高对抗性可移植性 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21181v1) [paper-pdf](http://arxiv.org/pdf/2505.21181v1)

**Authors**: Yayin Zheng, Chen Wan, Zihong Guo, Hailing Kuang, Xiaohai Lu

**Abstract**: Adversarial attacks have become a significant challenge in the security of machine learning models, particularly in the context of black-box defense strategies. Existing methods for enhancing adversarial transferability primarily focus on the spatial domain. This paper presents Frequency-Space Attack (FSA), a new adversarial attack framework that effectively integrates frequency-domain and spatial-domain transformations. FSA combines two key techniques: (1) High-Frequency Augmentation, which applies Fourier transform with frequency-selective amplification to diversify inputs and emphasize the critical role of high-frequency components in adversarial attacks, and (2) Hierarchical-Gradient Fusion, which merges multi-scale gradient decomposition and fusion to capture both global structures and fine-grained details, resulting in smoother perturbations. Our experiment demonstrates that FSA consistently outperforms state-of-the-art methods across various black-box models. Notably, our proposed FSA achieves an average attack success rate increase of 23.6% compared with BSR (CVPR 2024) on eight black-box defense models.

摘要: 对抗性攻击已经成为机器学习模型安全性的一个重大挑战，特别是在黑盒防御策略的背景下。用于增强对抗性可转移性的现有方法主要集中在空间域。本文提出了一种新的对抗性攻击框架--频率-空间攻击（FSA），它有效地集成了频域和空间域变换。FSA结合了两项关键技术：（1）高频增强，应用傅里叶变换和频率选择性放大来实现输入多样化，并强调高频分量在对抗攻击中的关键作用，（2）分层梯度融合，融合多尺度梯度分解和融合，以捕获全局结构和细粒度细节，从而产生更平滑的扰动。我们的实验表明，FSA在各种黑匣子模型中始终优于最先进的方法。值得注意的是，我们提出的FSA在八种黑匣子防御模型上的平均攻击成功率与SVR（CVPR 2024）相比提高了23.6%。



## **9. Unraveling Indirect In-Context Learning Using Influence Functions**

使用影响函数解开间接上下文学习 cs.LG

Under Review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.01473v2) [paper-pdf](http://arxiv.org/pdf/2501.01473v2)

**Authors**: Hadi Askari, Shivanshu Gupta, Terry Tong, Fei Wang, Anshuman Chhabra, Muhao Chen

**Abstract**: In this work, we introduce a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy ICL. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential of IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can further improve performance, leading to average absolute accuracy gains of 0.37\% and 1.45\% for 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy ICL setting, we examine scenarios where demonstrations might be mislabeled or have adversarial noise. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy by an average of 2.90\% for Cosine Similarity and 2.94\% for BSR on noisy GLUE benchmarks. For the adversarial sub-setting, we show the utility of using IFs for task-agnostic demonstration selection for backdoor attack mitigation. Showing a 32.89\% reduction in Attack Success Rate compared to task-aware methods. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL.

摘要: 在这项工作中，我们引入了一种新的广义内上下文学习（ICL）范式，称为间接内上下文学习。在间接ICL中，我们探索了为两种不同的现实世界场景量身定制的演示选择策略：任务混合和噪音ICL。我们系统地评估影响力函数（IF）作为这些环境的选择工具的有效性，强调了IF更好地捕捉演示池中示例信息量的潜力。对于“任务混合”设置，演示来自28个不同的任务，包括MMLU、BigBench、StrategyQA和CommonsenseQA。我们证明，将BertScore-Recall（SVR）与IF代理模型相结合可以进一步提高性能，与传统ICL指标相比，3次和5次设置的平均绝对准确性提高为0.37%和1.45%。在有噪音的ICL环境中，我们检查演示可能被错误标记或具有对抗性噪音的场景。我们的实验表明，用基于IF的选择器重新加权传统ICL选择器（BEP和Cosine相似性），在有噪的GLUE基准上，Cosine相似性的准确性平均提高了2.90%，而SVR的准确性平均提高了2.94%。对于对抗性子设置，我们展示了使用IF进行任务不可知演示选择以缓解后门攻击的实用性。与任务感知方法相比，攻击成功率降低了32.89%。总而言之，我们提出了一个强大的示范选择框架，该框架超越了传统ICL，为间接ICL中的国际单项指标的作用提供了有价值的见解。



## **10. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

TabAttackBench：表格数据对抗性攻击的基准 cs.LG

63 pages, 22 figures, 6 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21027v1) [paper-pdf](http://arxiv.org/pdf/2505.21027v1)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data.

摘要: 对抗性攻击通过对输入数据的不可感知的扰动引发错误预测，对机器学习模型构成重大威胁。虽然这些攻击在图像等非结构化数据中得到了广泛研究，但它们对表格数据的应用带来了新的挑战。这些挑战源于表格数据中固有的同质性和复杂的特征相互依赖性，而表格数据与图像数据中的数据有显着不同。为了解决这些差异，将不可感知性视为表格数据特定的关键标准至关重要。当前的大多数研究主要关注于实现有效的对抗攻击，而往往忽视了保持不可感知性的重要性。为了解决这一差距，我们提出了一个针对表格数据的对抗攻击的新基准，该基准评估有效性和不可感知性。在这项研究中，我们使用11个表格数据集（包括混合数据集和纯数字数据集）评估了四种模型中五种对抗攻击的有效性和不可感知性。我们的分析探讨了这些因素如何相互作用并影响攻击的整体性能。我们还比较了不同数据集类型的结果，以了解这些发现的更广泛影响。该基准的研究结果为改进对抗性攻击算法的设计提供了宝贵的见解，从而推进表格数据上的对抗性机器学习领域。



## **11. Tradeoffs Between Alignment and Helpfulness in Language Models with Steering Methods**

具有引导方法的语言模型中的一致性和帮助性之间的权衡 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2401.16332v5) [paper-pdf](http://arxiv.org/pdf/2401.16332v5)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

摘要: 语言模型对齐已成为人工智能安全的重要组成部分，通过增强期望的行为和抑制不期望的行为，允许人类与语言模型之间的安全交互。通常通过调整模型或插入预设对齐提示来完成。最近，表示工程（一种通过在训练后改变模型的表示来改变模型行为的方法）被证明在对齐LLM方面有效（Zou等人，2023 a）。表示工程在以对齐为导向的任务中产生了收益，例如抵抗对抗攻击和减少社会偏见，但也被证明会导致模型执行基本任务的能力下降。在本文中，我们研究了模型对齐度的增加和帮助性的减少之间的权衡。我们提出了一个理论框架，为这两个量提供了界限，并从经验上证明了它们的相关性。首先，我们发现在我们的框架条件下，可以通过表示工程来保证一致性，同时在这个过程中会损害帮助性。其次，我们表明，帮助性与表示工程载体的规范成二次关系受到损害，而对齐度则随其线性增加，这表明使用表示工程是有效的制度。我们从经验上验证了我们的发现，并绘制了对齐表示工程的有用性的界限。



## **12. NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion**

NatADiff：自然对抗扩散的对抗边界指南 cs.LG

10 pages, 3 figures, 2 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20934v1) [paper-pdf](http://arxiv.org/pdf/2505.20934v1)

**Authors**: Max Collins, Jordan Vice, Tim French, Ajmal Mian

**Abstract**: Adversarial samples exploit irregularities in the manifold ``learned'' by deep learning models to cause misclassifications. The study of these adversarial samples provides insight into the features a model uses to classify inputs, which can be leveraged to improve robustness against future attacks. However, much of the existing literature focuses on constrained adversarial samples, which do not accurately reflect test-time errors encountered in real-world settings. To address this, we propose `NatADiff', an adversarial sampling scheme that leverages denoising diffusion to generate natural adversarial samples. Our approach is based on the observation that natural adversarial samples frequently contain structural elements from the adversarial class. Deep learning models can exploit these structural elements to shortcut the classification process, rather than learning to genuinely distinguish between classes. To leverage this behavior, we guide the diffusion trajectory towards the intersection of the true and adversarial classes, combining time-travel sampling with augmented classifier guidance to enhance attack transferability while preserving image fidelity. Our method achieves comparable attack success rates to current state-of-the-art techniques, while exhibiting significantly higher transferability across model architectures and better alignment with natural test-time errors as measured by FID. These results demonstrate that NatADiff produces adversarial samples that not only transfer more effectively across models, but more faithfully resemble naturally occurring test-time errors.

摘要: 对抗性样本利用深度学习模型“学习”的多个分支中的不规则性来导致错误分类。对这些对抗性样本的研究可以深入了解模型用于分类输入的特征，这些特征可以用于提高针对未来攻击的鲁棒性。然而，大部分现有文献都集中在受约束的对抗样本上，这些样本并不能准确反映现实世界环境中遇到的测试时错误。为了解决这个问题，我们提出了“NatADiff”，这是一种对抗性采样方案，利用去噪扩散来生成自然对抗性样本。我们的方法基于这样的观察：自然对抗样本经常包含对抗类别的结构元素。深度学习模型可以利用这些结构元素来缩短分类过程，而不是学习真正区分类别。为了利用这种行为，我们将扩散轨迹引导到真实类和对抗类的交叉点，将时间旅行采样与增强分类器引导相结合，以增强攻击的可转移性，同时保持图像保真度。我们的方法实现了与当前最先进技术相当的攻击成功率，同时在模型架构中表现出显着更高的可移植性，并与FID测量的自然测试时间误差更好地对齐。这些结果表明，NatADiff产生的对抗样本不仅可以更有效地在模型之间传递，而且更忠实地类似于自然发生的测试时错误。



## **13. ZeroPur: Succinct Training-Free Adversarial Purification**

ZeroPur：简洁的免培训对抗净化 cs.CV

17 pages, 7 figures, under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2406.03143v3) [paper-pdf](http://arxiv.org/pdf/2406.03143v3)

**Authors**: Erhu Liu, Zonglin Yang, Bo Liu, Bin Xiao, Xiuli Bi

**Abstract**: Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.

摘要: 对抗净化是一种防御技术，可以在不修改受害者分类器的情况下防御各种看不见的对抗攻击。现有的方法通常依赖于外部生成模型或辅助功能和受害者分类器之间的合作。然而，重新训练生成模型、辅助函数或受害者分类器依赖于微调数据集的域，并且需要计算。在这项工作中，我们假设对抗图像是自然图像多维的离群值，净化过程可以被认为是将它们返回到该多维。遵循这一假设，我们提出了一种简单的对抗净化方法，无需进一步训练来净化对抗图像，称为ZeroPur。ZeroPur包含两个步骤：给定一个对抗性示例，Guided Change通过其模糊对应的引导获得对抗性示例的移动嵌入;之后，自适应投影通过这种移动嵌入来构建一个方向载体，以提供动量，将对抗性图像自适应地投影到多管上。ZeroPur独立于外部模型，不需要重新训练受害者分类器或辅助功能，仅依靠受害者分类器本身来实现净化。在三个数据集（CIFAR-10，CIFAR-100和ImageNet-1 K）上使用各种分类器架构（ResNet，WideResNet）进行的广泛实验表明，我们的方法实现了最先进的鲁棒性能。代码将公开。



## **14. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）的能力越来越强，对其安全部署的担忧也越来越大。尽管已经引入了对齐机制来阻止滥用，但它们仍然容易受到精心设计的对抗提示的影响。在这项工作中，我们提出了一种可扩展的攻击策略：意图隐藏对抗提示，通过技能的组合来隐藏恶意意图。我们开发了一个博弈论框架来建模此类攻击与应用提示和响应过滤的防御系统之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **15. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

MedSentry：了解和缓解医学LLM多主体系统中的安全风险 cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.

摘要: 随着大型语言模型（LLM）越来越多地部署在医疗保健中，确保其安全性，特别是在协作多代理配置中，至关重要。在本文中，我们介绍了MedSentry，这是一个基准，由5000个对抗性医疗提示组成，涵盖25个威胁类别和100个子主题。与此数据集相结合，我们开发了一个端到端的攻击防御评估管道，以系统地分析四种代表性的多智能体布局（Layers、SharedPool、Centralized和Decentralized）如何抵御来自“黑暗人格”智能体的攻击。我们的研究结果揭示了这些架构如何处理信息污染和维持稳健决策的关键差异，暴露了其潜在的脆弱性机制。例如，SharedPool的开放信息共享使其高度容易受到影响，而去中心化架构由于固有的冗余和隔离而表现出更大的弹性。为了减轻这些风险，我们提出了一种个性规模的检测和纠正机制，该机制可以识别和恢复恶意代理，将系统安全性恢复到接近基线的水平。因此，MedSentry提供了严格的评估框架和实用的防御策略，指导医学领域更安全的基于LLM的多智能体系统的设计。



## **16. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

TrojanStego：你的语言模型可以秘密地成为隐写隐私泄露代理 cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.

摘要: 随着大型语言模型（LLM）集成到敏感工作流程中，人们越来越担心它们泄露机密信息的可能性。我们提出了TrojanStego，这是一种新型威胁模型，其中对手微调LLM，通过语言隐写术将敏感的上下文信息嵌入到看起来自然的输出中，而不需要对推理输入进行显式控制。我们引入了一个分类法，概述了受影响的LLM的风险因素，并使用它来评估威胁的风险状况。为了实现TrojanStego，我们提出了一种基于词汇划分的实用编码方案，LLM可以通过微调学习。实验结果表明，受攻击的模型在发出的提示上以87%的准确率可靠地传输32位秘密，使用三代多数投票的准确率达到97%以上。此外，它们保持高实用性，可以逃避人类检测，并保持一致性。这些结果凸显了一类新型LLM数据泄露攻击，这些攻击是被动的、隐蔽的、实用的且危险的。



## **17. Breaking Dataset Boundaries: Class-Agnostic Targeted Adversarial Attacks**

突破数据集边界：类别不可知的针对性对抗攻击 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20782v1) [paper-pdf](http://arxiv.org/pdf/2505.20782v1)

**Authors**: Taïga Gonçalves, Tomo Miyazaki, Shinichiro Omachi

**Abstract**: We present Cross-Domain Multi-Targeted Attack (CD-MTA), a method for generating adversarial examples that mislead image classifiers toward any target class, including those not seen during training. Traditional targeted attacks are limited to one class per model, requiring expensive retraining for each target. Multi-targeted attacks address this by introducing a perturbation generator with a conditional input to specify the target class. However, existing methods are constrained to classes observed during training and require access to the black-box model's training data--introducing a form of data leakage that undermines realistic evaluation in practical black-box scenarios. We identify overreliance on class embeddings as a key limitation, leading to overfitting and poor generalization to unseen classes. To address this, CD-MTA replaces class-level supervision with an image-based conditional input and introduces class-agnostic losses that align the perturbed and target images in the feature space. This design removes dependence on class semantics, thereby enabling generalization to unseen classes across datasets. Experiments on ImageNet and seven other datasets show that CD-MTA outperforms prior multi-targeted attacks in both standard and cross-domain settings--without accessing the black-box model's training data.

摘要: 我们提出了跨域多目标攻击（CD-MTA），这是一种生成对抗性示例的方法，可以将图像分类器误导到任何目标类别，包括训练期间未看到的目标类别。传统的有针对性的攻击每个模型仅限于一个类别，需要对每个目标进行昂贵的再培训。多目标攻击通过引入带有条件输入的扰动生成器来解决这个问题以指定目标类。然而，现有的方法局限于训练期间观察到的类，并且需要访问黑匣子模型的训练数据--引入了一种形式的数据泄露，破坏了实际黑匣子场景中的现实评估。我们认为过度依赖类嵌入是一个关键限制，导致对未见类的过度适应和较差的概括。为了解决这个问题，CD-MTA用基于图像的条件输入取代了类级监督，并引入了类不可知损失，使特征空间中的受扰动图像和目标图像对齐。这种设计消除了对类语义的依赖，从而能够在数据集中推广到看不见的类。ImageNet和其他七个数据集的实验表明，CD-MTA在标准和跨域设置中都优于之前的多目标攻击，而无需访问黑匣子模型的训练数据。



## **18. SC-Pro: Training-Free Framework for Defending Unsafe Image Synthesis Attack**

SC-Pro：用于防御不安全图像合成攻击的免培训框架 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.05359v2) [paper-pdf](http://arxiv.org/pdf/2501.05359v2)

**Authors**: Junha Park, Jaehui Hwang, Ian Ryu, Hyungkeun Park, Jiyoon Kim, Jong-Seok Lee

**Abstract**: With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose SC-Pro (Spherical or Circular Probing), a training-free framework that easily defends against adversarial attacks generating NSFW images. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (SC-Pro-o), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.

摘要: 随着扩散模型的进步，图像生成的性能得到了显着的改进。这引发了人们对图像生成潜在滥用的担忧，例如创建露骨或暴力图像，通常称为“不安全工作（NSFW）内容”。为了解决这个问题，稳定扩散模型包括几个安全检查器，以审查从模型生成的初始文本提示和最终输出图像。然而，最近的研究表明，这些安全检查器具有对抗性攻击的漏洞，使它们能够生成NSFW图像。在本文中，我们发现这些对抗攻击对文本提示或输入潜伏的微小变化并不稳健。基于此，我们提出了SC-Pro（球形或圆形探测），这是一个免训练框架，可以轻松防御生成NSFW图像的对抗攻击。此外，我们开发了一种利用一步扩散模型进行高效NSFW检测（SC-Pro-o）的方法，进一步减少了计算资源。我们证明了我们的方法在性能和适用性方面的优越性。



## **19. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

迈向LLM摆脱学习对重新学习攻击的弹性：敏锐意识的最小化视角及超越 cs.LG

Accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.05374v4) [paper-pdf](http://arxiv.org/pdf/2502.05374v4)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.

摘要: 最近引入了LLM取消学习技术，以遵守数据法规，并通过消除不希望的数据模型影响来解决LLM的安全和道德问题。然而，最先进的忘记方法面临着一个严重的漏洞：它们容易受到从少数忘记数据点中“重新学习”删除的信息的影响，称为重新学习攻击。在本文中，我们系统地研究了如何使未学习的模型对此类攻击具有鲁棒性。我们首次通过统一的鲁棒优化框架在鲁棒取消学习和敏锐度感知最小化（Sam）之间建立了联系，类似于旨在防御对抗攻击的对抗训练。我们对Sam的分析表明，平滑度优化在减轻重新学习攻击方面发挥着关键作用。因此，我们进一步探索不同的平滑策略以增强取消学习鲁棒性。对WMDP和MUSE等基准数据集的广泛实验表明，Sam和其他平滑度优化方法能够持续提高LLM取消学习对重新学习攻击的抵抗力。值得注意的是，平滑性增强的忘记学习还有助于抵御（输入级）越狱攻击，扩大我们提案在增强LLM忘记学习方面的影响。代码可访问https://github.com/OPTML-Group/Unlearn-Smooth。



## **20. Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning**

离线强化学习中针对中毒攻击的多级别认证防御 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20621v1) [paper-pdf](http://arxiv.org/pdf/2505.20621v1)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein

**Abstract**: Similar to other machine learning frameworks, Offline Reinforcement Learning (RL) is shown to be vulnerable to poisoning attacks, due to its reliance on externally sourced datasets, a vulnerability that is exacerbated by its sequential nature. To mitigate the risks posed by RL poisoning, we extend certified defenses to provide larger guarantees against adversarial manipulation, ensuring robustness for both per-state actions, and the overall expected cumulative reward. Our approach leverages properties of Differential Privacy, in a manner that allows this work to span both continuous and discrete spaces, as well as stochastic and deterministic environments -- significantly expanding the scope and applicability of achievable guarantees. Empirical evaluations demonstrate that our approach ensures the performance drops to no more than $50\%$ with up to $7\%$ of the training data poisoned, significantly improving over the $0.008\%$ in prior work~\citep{wu_copa_2022}, while producing certified radii that is $5$ times larger as well. This highlights the potential of our framework to enhance safety and reliability in offline RL.

摘要: 与其他机器学习框架类似，离线强化学习（RL）由于依赖于外部来源的数据集，容易受到中毒攻击，这种脆弱性因其顺序性而加剧。为了减轻RL中毒带来的风险，我们扩展了经过认证的防御，以提供更大的对抗性操纵的保证，确保每个状态操作和整体预期累积奖励的鲁棒性。我们的方法利用差分隐私的属性，以允许这项工作跨越连续和离散空间以及随机和确定性环境的方式-显着扩展可实现保证的范围和适用性。经验评估表明，我们的方法确保性能下降到不超过$50\%$与高达$7\%$的训练数据中毒，显着提高了$0.008\%$在以前的工作~\citep{wu_copa_2022}，同时产生认证的半径是$5$倍大，以及。这凸显了我们的框架增强离线RL安全性和可靠性的潜力。



## **21. Detector noise in continuous-variable quantum key distribution**

连续变量量子密钥分配中的检测器噪音 quant-ph

7 pages, 6 figures

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20441v1) [paper-pdf](http://arxiv.org/pdf/2505.20441v1)

**Authors**: Shihong Pan, Dimitri Monokandylos, Bing Qi

**Abstract**: Detector noise is a critical factor in determining the performance of a quantum key distribution (QKD) system. In continuous-variable (CV) QKD with optical coherent detection, the trusted detector noise model is widely used to enhance both the secret key rate and transmission distance. This model assumes that noise from the coherent detector is inherently random and cannot be accessed or manipulated by an adversary. Its validity rests on two key assumptions: (1) the detector can be accurately calibrated by the legitimate user and remains isolated from the adversary, and (2) the detector noise is truly random. So far, extensive research has focused on detector calibration and countermeasures against detector side-channel attacks. However, there is no strong evidence supporting assumption (2). In this paper, we analyze the electrical noise of a commercial balanced Photoreceiver, which has been applied in CV-QKD implementations, and demonstrate that assumption (2) is unjustified. To address this issue, we propose a "calibrated detector noise" model for CV-QKD, which relies solely on assumption (1). Numerical simulations comparing different noise models indicate that the new model can achieve a secret key rate comparable to the trusted-noise model, without depending on the questionable assumption of "truly random" detector noise.

摘要: 检测器噪音是决定量子密钥分发（QKD）系统性能的关键因素。在具有光相关检测的连续变量（CV）QKD中，可信检测器噪音模型被广泛用于提高密钥速率和传输距离。该模型假设来自相关检测器的噪音本质上是随机的，并且无法被对手访问或操纵。其有效性取决于两个关键假设：（1）检测器可以由合法用户准确校准并与对手保持隔离，（2）检测器噪音确实是随机的。到目前为止，广泛的研究集中在检测器校准和针对检测器侧通道攻击的对策上。然而，没有强有力的证据支持假设（2）。在本文中，我们分析了已应用于CV-QKD实现的商用平衡光接收器的电噪音，并证明假设（2）是不合理的。为了解决这个问题，我们提出了CV-QKD的“校准检测器噪音”模型，该模型仅依赖于假设（1）。数值模拟比较不同的噪声模型表明，新的模型可以实现一个秘密的密钥速率相媲美的可信噪声模型，而不依赖于可疑的假设“真正随机”的检测器噪声。



## **22. Holes in Latent Space: Topological Signatures Under Adversarial Influence**

潜在空间中的漏洞：敌对影响下的布局特征 cs.LG

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20435v1) [paper-pdf](http://arxiv.org/pdf/2505.20435v1)

**Authors**: Aideen Fay, Inés García-Redondo, Qiquan Wang, Haim Dubossarsky, Anthea Monod

**Abstract**: Understanding how adversarial conditions affect language models requires techniques that capture both global structure and local detail within high-dimensional activation spaces. We propose persistent homology (PH), a tool from topological data analysis, to systematically characterize multiscale latent space dynamics in LLMs under two distinct attack modes -- backdoor fine-tuning and indirect prompt injection. By analyzing six state-of-the-art LLMs, we show that adversarial conditions consistently compress latent topologies, reducing structural diversity at smaller scales while amplifying dominant features at coarser ones. These topological signatures are statistically robust across layers, architectures, model sizes, and align with the emergence of adversarial effects deeper in the network. To capture finer-grained mechanisms underlying these shifts, we introduce a neuron-level PH framework that quantifies how information flows and transforms within and across layers. Together, our findings demonstrate that PH offers a principled and unifying approach to interpreting representational dynamics in LLMs, particularly under distributional shift.

摘要: 了解对抗条件如何影响语言模型需要捕获多维激活空间内的全局结构和局部细节的技术。我们提出了持久同调（PH），这是一种来自拓扑数据分析的工具，用于系统地描述两种不同攻击模式（后门微调和间接即时注入）下LLM中的多尺度潜在空间动力学。通过分析六种最先进的LLM，我们表明对抗条件会持续压缩潜在布局，减少较小规模的结构多样性，同时放大较粗规模的主导特征。这些拓扑签名在各个层、架构、模型大小上都具有统计稳健性，并与网络中更深层次对抗效应的出现保持一致。为了捕捉这些转变背后的细粒度机制，我们引入了一个神经元级PH框架，该框架量化信息如何在层内和层之间流动和转换。总的来说，我们的研究结果表明，PH提供了一种有原则且统一的方法来解释法学硕士中的代表性动态，特别是在分布转变下。



## **23. Eradicating the Unseen: Detecting, Exploiting, and Remediating a Path Traversal Vulnerability across GitHub**

消除隐形：检测、利用和修复GitHub上的路径穿越漏洞 cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20186v1) [paper-pdf](http://arxiv.org/pdf/2505.20186v1)

**Authors**: Jafar Akhoundali, Hamidreza Hamidi, Kristian Rietveld, Olga Gadyatskaya

**Abstract**: Vulnerabilities in open-source software can cause cascading effects in the modern digital ecosystem. It is especially worrying if these vulnerabilities repeat across many projects, as once the adversaries find one of them, they can scale up the attack very easily. Unfortunately, since developers frequently reuse code from their own or external code resources, some nearly identical vulnerabilities exist across many open-source projects.   We conducted a study to examine the prevalence of a particular vulnerable code pattern that enables path traversal attacks (CWE-22) across open-source GitHub projects. To handle this study at the GitHub scale, we developed an automated pipeline that scans GitHub for the targeted vulnerable pattern, confirms the vulnerability by first running a static analysis and then exploiting the vulnerability in the context of the studied project, assesses its impact by calculating the CVSS score, generates a patch using GPT-4, and reports the vulnerability to the maintainers.   Using our pipeline, we identified 1,756 vulnerable open-source projects, some of which are very influential. For many of the affected projects, the vulnerability is critical (CVSS score higher than 9.0), as it can be exploited remotely without any privileges and critically impact the confidentiality and availability of the system. We have responsibly disclosed the vulnerability to the maintainers, and 14\% of the reported vulnerabilities have been remediated.   We also investigated the root causes of the vulnerable code pattern and assessed the side effects of the large number of copies of this vulnerable pattern that seem to have poisoned several popular LLMs. Our study highlights the urgent need to help secure the open-source ecosystem by leveraging scalable automated vulnerability management solutions and raising awareness among developers.

摘要: 开源软件中的漏洞可能会在现代数字生态系统中造成连锁效应。如果这些漏洞在许多项目中重复出现，尤其令人担忧，因为一旦对手发现其中一个漏洞，他们就可以很容易地扩大攻击规模。不幸的是，由于开发人员经常重复使用自己或外部代码资源中的代码，因此许多开源项目中都存在一些几乎相同的漏洞。   我们进行了一项研究来检查特定易受攻击代码模式的普遍性，该模式导致开源GitHub项目中的路径穿越攻击（CWE-22）。为了以GitHub规模处理这项研究，我们开发了一个自动化管道，该管道扫描GitHub的目标漏洞模式，通过首先运行静态分析来确认漏洞，然后在研究项目的上下文中利用漏洞，通过计算CVD评分来评估其影响，使用GPT-4生成补丁，并向维护者报告漏洞。   使用我们的管道，我们发现了1，756个脆弱的开源项目，其中一些非常有影响力。对于许多受影响的项目来说，该漏洞至关重要（CVD评分高于9.0），因为它可以在没有任何特权的情况下被远程利用，并严重影响系统的机密性和可用性。我们已负责任地向维护者披露了该漏洞，报告的漏洞中有14%已得到修复。   我们还调查了易受攻击的代码模式的根本原因，并评估了该易受攻击模式的大量副本的副作用，这些副本似乎已经毒害了几种流行的LLM。我们的研究强调，迫切需要通过利用可扩展的自动化漏洞管理解决方案和提高开发人员的意识来帮助保护开源生态系统。



## **24. On the Robustness of RSMA to Adversarial BD-RIS-Induced Interference**

RMA对对抗性BD-RIS诱导干扰的鲁棒性 eess.SP

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20146v1) [paper-pdf](http://arxiv.org/pdf/2505.20146v1)

**Authors**: Arthur S. de Sena, Jacek Kibilda, Nurul H. Mahmood, Andre Gomes, Luiz A. DaSilva, Matti Latva-aho

**Abstract**: This article investigates the robustness of rate-splitting multiple access (RSMA) in multi-user multiple-input multiple-output (MIMO) systems to interference attacks against channel acquisition induced by beyond-diagonal RISs (BD-RISs). Two primary attack strategies, random and aligned interference, are proposed for fully connected and group-connected BD-RIS architectures. Valid random reflection coefficients are generated exploiting the Takagi factorization, while potent aligned interference attacks are achieved through optimization strategies based on a quadratically constrained quadratic program (QCQP) reformulation followed by projections onto the unitary manifold. Our numerical findings reveal that, when perfect channel state information (CSI) is available, RSMA behaves similarly to space-division multiple access (SDMA) and thus is highly susceptible to the attack, with BD-RIS inducing severe performance loss and significantly outperforming diagonal RIS. However, under imperfect CSI, RSMA consistently demonstrates significantly greater robustness than SDMA, particularly as the system's transmit power increases.

摘要: 本文研究了多用户多输入多输出（MIMO）系统中的速率分裂多址（RSMA）对超对角RIS（BD-RIS）引起的针对信道捕获的干扰攻击的鲁棒性。针对全连接和群连接的BD-RIS架构提出了两种主要的攻击策略：随机干扰和对齐干扰。有效的随机反射系数是利用Takagi因式分解生成的，而有效的对齐干扰攻击是通过基于二次约束二次规划（QCQP）重新公式的优化策略来实现的，然后是投影到正总管上。我们的数字研究结果表明，当完美通道状态信息（SI）可用时，RMA的行为与空间分多址（SBA）类似，因此极易受到攻击，其中BD-RIS会导致严重的性能损失，并且显着优于对角RIS。然而，在不完美的SI下，RMA始终表现出比SBA明显更高的鲁棒性，特别是当系统发射功率增加时。



## **25. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.13862v3) [paper-pdf](http://arxiv.org/pdf/2505.13862v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **26. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.05050v3) [paper-pdf](http://arxiv.org/pdf/2504.05050v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **27. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2501.06044v6) [paper-pdf](http://arxiv.org/pdf/2501.06044v6)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议都可以实现的最佳故障容差在各种设置中都有其特征。例如，对于在部分同步设置中运行的状态机复制（SVR）协议，可以同时保证针对$\Alpha$-有界对手（即，控制少于参与者$\Alpha$一部分的对手）和针对$\Beta$的活力-有界的对手当且仅当$\Alpha +2\Beta\leq 1$。   本文描述了当放宽标准一致性要求以允许有界数量$r$的一致性违规时，SVR协议在多大程度上可能实现“优于最优”的公差保证。我们证明，如果没有额外的时间假设，限制回滚是不可能的，并研究每当攻击时间左右的消息延迟受到参数$\Delta '*$（该参数可以任意大于部分同步模型中限制后GST消息延迟的参数$\Delta$）时，能够容忍一致性违规并从一致性违规中恢复的协议。在这里，协议的故障容限可以是$r$的非常函数，并且我们证明，对于每个$r$，任何SVR协议可实现的最佳“可恢复故障容限”的上下限和下限匹配。例如，对于在部分同步设置中保证针对1/3有界对手的活性的协议，5/9有界对手总是会导致一次一致性违规，但不会导致两次一致性违规，而2/3有界对手总是会导致两次一致性违规，但不会导致三次。我们的积极结果是通过通用的“恢复程序”实现的，该程序可以移植到任何负责任的SVR协议上，并在违规后恢复一致性，同时仅回滚在之前$2\Delta '*$时间步中完成的事务。



## **28. Attention! You Vision Language Model Could Be Maliciously Manipulated**

注意！您的视觉语言模型可能被恶意操纵 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19911v1) [paper-pdf](http://arxiv.org/pdf/2505.19911v1)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.

摘要: 大型视觉语言模型（VLM）在理解复杂的现实世界场景和支持数据驱动的决策流程方面取得了显着的成功。然而，VLM对对抗性示例（无论是文本还是图像）表现出显着的脆弱性，这可能会导致各种对抗性结果，例如越狱、劫持和幻觉等。在这项工作中，我们从经验和理论上证明了VLM特别容易受到基于图像的对抗示例的影响，其中不可感知的扰动可以精确地操纵每个输出令牌。为此，我们提出了一种名为视觉语言模型操纵攻击（VMA）的新型攻击，该攻击将一阶和二阶动量优化技术与可微转换机制集成在一起，以有效地优化对抗性扰动。值得注意的是，VMA可以是一把双刃剑：它可以被用来实施各种攻击，例如越狱、劫持、隐私泄露、拒绝服务和海绵示例的生成等，同时允许注入水印以进行版权保护。广泛的实证评估证实了VMA在不同场景和数据集中的有效性和普遍性。



## **29. CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models**

CPA-RAG：对大型语言模型中检索增强生成的隐蔽中毒攻击 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19864v1) [paper-pdf](http://arxiv.org/pdf/2505.19864v1)

**Authors**: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but its openness introduces vulnerabilities that can be exploited by poisoning attacks. Existing poisoning methods for RAG systems have limitations, such as poor generalization and lack of fluency in adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial framework that generates query-relevant texts capable of manipulating the retrieval process to induce target answers. The proposed method integrates prompt-based text generation, cross-guided optimization through multiple LLMs, and retriever-based scoring to construct high-quality adversarial samples. We conduct extensive experiments across multiple datasets and LLMs to evaluate its effectiveness. Results show that the framework achieves over 90\% attack success when the top-k retrieval setting is 5, matching white-box performance, and maintains a consistent advantage of approximately 5 percentage points across different top-k values. It also outperforms existing black-box baselines by 14.5 percentage points under various defense strategies. Furthermore, our method successfully compromises a commercial RAG system deployed on Alibaba's BaiLian platform, demonstrating its practical threat in real-world applications. These findings underscore the need for more robust and secure RAG frameworks to defend against poisoning attacks.

摘要: 检索增强生成（RAG）通过合并外部知识来增强大型语言模型（LLM），但其开放性引入了可被中毒攻击利用的漏洞。现有的RAG系统中毒方法存在局限性，例如概括性较差以及对抗性文本缺乏流畅性。在本文中，我们提出了CPA-RAG，这是一个黑盒对抗框架，可以生成与查询相关的文本，这些文本能够操纵检索过程以诱导目标答案。所提出的方法集成了基于文本的生成，通过多个LLM的交叉引导优化，以及基于检索器的评分来构建高质量的对抗样本。我们在多个数据集和LLM中进行了广泛的实验，以评估其有效性。结果表明，当top-k检索设置为5时，该框架的攻击成功率超过90%，与白盒性能相匹配，并在不同top-k值之间保持约5个百分点的一致优势。在各种防御策略下，它还比现有黑匣子基线高出14.5个百分点。此外，我们的方法成功地破坏了阿里巴巴百联平台上部署的商业RAG系统，证明了其在现实应用中的实际威胁。这些发现强调了需要更强大、更安全的RAG框架来抵御中毒攻击。



## **30. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

越狱提示攻击：针对扩散模型的可控对抗攻击 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2404.02928v4) [paper-pdf](http://arxiv.org/pdf/2404.02928v4)

**Authors**: Jiachen Ma, Yijiang Li, Zhiqing Xiao, Anda Cao, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: Text-to-image (T2I) models can be maliciously used to generate harmful content such as sexually explicit, unfaithful, and misleading or Not-Safe-for-Work (NSFW) images. Previous attacks largely depend on the availability of the diffusion model or involve a lengthy optimization process. In this work, we investigate a more practical and universal attack that does not require the presence of a target model and demonstrate that the high-dimensional text embedding space inherently contains NSFW concepts that can be exploited to generate harmful images. We present the Jailbreaking Prompt Attack (JPA). JPA first searches for the target malicious concepts in the text embedding space using a group of antonyms generated by ChatGPT. Subsequently, a prefix prompt is optimized in the discrete vocabulary space to align malicious concepts semantically in the text embedding space. We further introduce a soft assignment with gradient masking technique that allows us to perform gradient ascent in the discrete vocabulary space.   We perform extensive experiments with open-sourced T2I models, e.g. stable-diffusion-v1-4 and closed-sourced online services, e.g. DALLE2, Midjourney with black-box safety checkers. Results show that (1) JPA bypasses both text and image safety checkers (2) while preserving high semantic alignment with the target prompt. (3) JPA demonstrates a much faster speed than previous methods and can be executed in a fully automated manner. These merits render it a valuable tool for robustness evaluation in future text-to-image generation research.

摘要: 文本到图像（T2 I）模型可能被恶意用于生成有害内容，例如露骨的性内容、不忠实的、误导性的或不安全的工作（NSFW）图像。之前的攻击很大程度上取决于扩散模型的可用性或涉及漫长的优化过程。在这项工作中，我们研究了一种更实用、更通用的攻击，它不需要目标模型的存在，并证明了多维文本嵌入空间固有地包含可以被利用来生成有害图像的NSFW概念。我们介绍越狱提示攻击（JPA）。JPA首先使用ChatGPT生成的一组反例在文本嵌入空间中搜索目标恶意概念。随后，在离散词汇空间中优化前置提示，以在文本嵌入空间中对恶意概念进行语义对齐。我们进一步引入了一种具有梯度掩蔽技术的软分配，使我们能够在离散词汇空间中执行梯度上升。   我们使用开源T2 I模型（例如稳定扩散-v1-4）和封闭源在线服务（例如DALLE 2、带有黑匣子安全检查器的Midjourney）进行了广泛的实验。结果表明：（1）JPA绕过了文本和图像安全检查器（2），同时保持与目标提示的高度语义一致。(3)JPA的速度比以前的方法快得多，并且可以以完全自动化的方式执行。这些优点使其成为未来文本到图像生成研究中稳健性评估的宝贵工具。



## **31. One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP**

愚弄所有人的一个代理人：CLIP的普遍、可转移和有针对性的对抗攻击 cs.CR

21 pages, 15 figures, 18 tables To appear in the Proceedings of The  ACM Conference on Computer and Communications Security (CCS), 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19840v1) [paper-pdf](http://arxiv.org/pdf/2505.19840v1)

**Authors**: Binyan Xu, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) have achieved widespread success yet remain prone to adversarial attacks. Typically, such attacks either involve frequent queries to the target model or rely on surrogate models closely mirroring the target model -- often trained with subsets of the target model's training data -- to achieve high attack success rates through transferability. However, in realistic scenarios where training data is inaccessible and excessive queries can raise alarms, crafting adversarial examples becomes more challenging. In this paper, we present UnivIntruder, a novel attack framework that relies solely on a single, publicly available CLIP model and publicly available datasets. By using textual concepts, UnivIntruder generates universal, transferable, and targeted adversarial perturbations that mislead DNNs into misclassifying inputs into adversary-specified classes defined by textual concepts.   Our extensive experiments show that our approach achieves an Attack Success Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly outperforming existing transfer-based methods. Additionally, we reveal real-world vulnerabilities, showing that even without querying target models, UnivIntruder compromises image search engines like Google and Baidu with ASR rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR rates up to 80%. These findings underscore the practicality of our attack in scenarios where traditional avenues are blocked, highlighting the need to reevaluate security paradigms in AI applications.

摘要: 深度神经网络（DNN）已取得广泛成功，但仍然容易受到对抗攻击。通常，此类攻击要么涉及对目标模型的频繁查询，要么依赖于密切反映目标模型的代理模型（通常使用目标模型训练数据的子集进行训练），通过可移植性实现高攻击成功率。然而，在训练数据不可访问且过多查询可能引发警报的现实场景中，制作对抗性示例变得更具挑战性。在本文中，我们介绍了UnivInsurder，这是一种新型攻击框架，仅依赖于单个公开可用的CLIP模型和公开可用的数据集。通过使用文本概念，UnivInvurder生成普遍的、可转移的和有针对性的对抗性扰动，这些扰动误导DNN将输入错误分类到由文本概念定义的对抗指定的类中。   我们广泛的实验表明，我们的方法在ImageNet上实现了高达85%的攻击成功率（ASB），在CIFAR-10上实现了超过99%的攻击成功率，显着优于现有的基于传输的方法。此外，我们还揭示了现实世界的漏洞，表明即使不查询目标模型，UnivInsurder也会损害Google和Baidu等图像搜索引擎的ASB率高达84%，以及GPT-4和Claude-3.5等视觉语言模型的ASB率高达80%。这些发现强调了我们在传统途径被封锁的情况下攻击的实用性，强调了重新评估人工智能应用程序中安全范式的必要性。



## **32. TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization**

TESSER：通过频谱和语义正规化来自视觉变形者的传输增强对抗攻击 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19613v1) [paper-pdf](http://arxiv.org/pdf/2505.19613v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER} -- a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 12 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.

摘要: 对抗性可移植性仍然是评估深度神经网络稳健性的一个关键挑战。在安全关键型应用程序中，可移植性可以在不访问模型内部的情况下进行黑匣子攻击，使其成为现实世界对抗威胁评估的关键问题。虽然Vision Transformers（ViT）表现出了强大的对抗性能，但现有的攻击往往无法有效地跨架构转移，特别是从ViT到卷积神经网络（CNN）或混合模型。在本文中，我们介绍了\textBF{TESSER} --一种新型的对抗攻击框架，通过两个关键策略增强可移植性：（1）\textit{条件敏感的梯度缩放（FMSG）}，它根据从中间特征激活中获得的标记重要性来调制梯度，和（2）\textit{光谱平滑度正规化（SSSR）}，它使用可微高斯先验来抑制扰动中的高频噪音。这些组件协同工作，以生成语义上有意义且频谱上光滑的扰动。在ImageNet上跨12种不同架构的广泛实验表明，与最先进的自适应令牌调整（ATA）方法相比，TESSER在CNN上的攻击成功率（ASB）提高了+10.9%，ViT上的攻击成功率（ASB）提高了+7.2%。此外，TESSER显着提高了针对受保护模型的鲁棒性，在对抗训练的CNN上实现了53.55%ASB。定性分析显示TESSER的扰动与通过Grad-CAM识别的突出视觉区域之间存在很强的一致性，而频域分析显示高频能量减少了12%，证实了频谱规则化的有效性。



## **33. JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models**

越狱：打破视觉语言模型的内部安全边界 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19610v1) [paper-pdf](http://arxiv.org/pdf/2505.19610v1)

**Authors**: Jiaxin Song, Yixu Wang, Jie Li, Rui Yu, Yan Teng, Xingjun Ma, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.

摘要: 视觉语言模型（VLM）表现出令人印象深刻的性能，但强大的视觉编码器的集成显着拓宽了它们的攻击面，使它们越来越容易受到越狱攻击。然而，由于缺乏明确定义的攻击目标，现有的越狱方法常常难以应对基于梯度的策略，这些策略容易出现局部最优情况，并且缺乏精确的方向指导，并且通常会使视觉和文本模式脱钩，从而通过忽视关键的跨模式交互来限制其有效性。受启发潜在知识（ELK）框架的启发，我们推测VLM在其内部融合层表示中编码安全相关信息，揭示了潜在空间中的隐性安全决策边界。这激励了利用边界来引导模型行为。因此，我们提出了JailBound，这是一种新型的潜在空间越狱框架，包括两个阶段：（1）安全边界探测，通过逼近融合层潜在空间内的决策边界来解决引导问题，从而识别朝向目标区域的最佳扰动方向;及（2）安全过境，它通过联合优化图像和文本输入的对抗性扰动来克服脱钩方法的局限性。后一阶段采用创新机制来引导模型的内部状态转向违反政策的输出，同时保持跨模式语义一致性。在六种不同的VLM上进行了大量实验，证明了JailBound的有效性，平均白盒攻击成功率为94.32%，黑盒攻击成功率为67.28%，分别比SOTA方法高6.17%和21.13%。我们的研究结果揭示了VLM中被忽视的安全风险，并强调了对更强大防御的迫切需要。警告：本文包含潜在敏感、有害和冒犯性内容。



## **34. Authenticated Sublinear Quantum Private Information Retrieval**

认证的亚线性量子私有信息检索 quant-ph

11 pages, 1 figure

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.04041v2) [paper-pdf](http://arxiv.org/pdf/2504.04041v2)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.

摘要: 本文使用量子相对信息引入了一种新颖的通信复杂性下界，完善了之前经典的基于信息量的结果。通过利用乌尔曼引理和量子平斯克不等式，作者为信息论安全建立了更严格的界限，证明量子协议在平衡隐私和效率方面本质上优于经典协议。还探讨了对称量子私有信息检索（QPIR）协议，该协议可以实现次线性通信复杂性，同时确保针对似是而非的对手的鲁棒性：一种基于后量子密码学的协议，可以为似是而非的服务器进行身份验证;一种基于环LWE的协议，用于单服务器设置中的后量子安全，确保针对量子攻击的鲁棒性;针对硬件实用性进行了优化的多服务器协议，在保持次线性效率的同时减少了实施费用。这些协议解决了安全数据库查询中的关键漏洞，提供了比经典线性复杂性方法呈指数级的通信改进。该工作还分析了量子似是而非的对手下的安全权衡，为隐私和正确性提供理论保证。



## **35. RDI: An adversarial robustness evaluation metric for deep neural networks based on model statistical features**

RDI：基于模型统计特征的深度神经网络对抗稳健性评估指标 cs.LG

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.18556v2) [paper-pdf](http://arxiv.org/pdf/2504.18556v2)

**Authors**: Jialei Song, Xingquan Zuo, Feiyang Wang, Hai Huang, Tianle Zhang

**Abstract**: Deep neural networks (DNNs) are highly susceptible to adversarial samples, raising concerns about their reliability in safety-critical tasks. Currently, methods of evaluating adversarial robustness are primarily categorized into attack-based and certified robustness evaluation approaches. The former not only relies on specific attack algorithms but also is highly time-consuming, while the latter due to its analytical nature, is typically difficult to implement for large and complex models. A few studies evaluate model robustness based on the model's decision boundary, but they suffer from low evaluation accuracy. To address the aforementioned issues, we propose a novel adversarial robustness evaluation metric, Robustness Difference Index (RDI), which is based on model statistical features. RDI draws inspiration from clustering evaluation by analyzing the intra-class and inter-class distances of feature vectors separated by the decision boundary to quantify model robustness. It is attack-independent and has high computational efficiency. Experiments show that, RDI demonstrates a stronger correlation with the gold-standard adversarial robustness metric of attack success rate (ASR). The average computation time of RDI is only 1/30 of the evaluation method based on the PGD attack. Our open-source code is available at: https://github.com/BUPTAIOC/RDI.

摘要: 深度神经网络（DNN）极易受到对抗样本的影响，这引发了人们对其在安全关键任务中可靠性的担忧。目前，评估对抗稳健性的方法主要分为基于攻击的稳健性评估方法和经过认证的稳健性评估方法。前者不仅依赖于特定的攻击算法，而且非常耗时，而后者由于其分析性质，通常很难对大型和复杂的模型实施。一些研究根据模型的决策边界评估模型稳健性，但评估准确性较低。为了解决上述问题，我们提出了一种新型的对抗鲁棒性评估指标--鲁棒性差异指数（RDI），它基于模型统计特征。RDI通过分析由决策边界分离的特征载体的类内和类间距离来量化模型稳健性，从集群评估中汲取灵感。它与攻击无关，计算效率高。实验表明，RDI与攻击成功率（SVR）的金标准对抗鲁棒性指标表现出更强的相关性。RDI的平均计算时间仅为基于PVD攻击的评估方法的1/30。我们的开源代码可访问：www.example.com。



## **36. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

一次性即可：将多回合攻击整合为LLM的高效单回合攻击 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2503.04856v2) [paper-pdf](http://arxiv.org/pdf/2503.04856v2)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.

摘要: 我们引入了一种新颖的框架，用于将多轮对抗性“越狱”提示整合到单轮查询中，从而显着减少了大型语言模型（LLM）对抗性测试所需的手动负担。虽然多回合人类越狱已被证明具有很高的攻击成功率，但它们需要相当大的人力和时间。我们的多回合到单回合（M2 S）方法--连字符化、数字化和Python化--系统地将多回合对话重新格式化为结构化的单回合提示。尽管消除了迭代的来回相互作用，但这些提示仍然保留并经常增强对抗能力：在对多回合人类越狱（MHJ）数据集的广泛评估中，M2 S方法在几种最先进的LLM中实现了从70.6%到95.9%的攻击成功率。值得注意的是，单回合提示的性能比最初的多回合攻击高出17.5个百分点，同时平均将代币使用量减少一半以上。进一步的分析表明，将恶意请求嵌入到列举或类代码结构中利用了“上下文盲目性”，绕过了本地护栏和外部输入输出过滤器。通过将多回合对话转换为简洁的单回合提示，M2 S框架为大规模红色团队提供了可扩展的工具，并揭示了当代LLM防御中的关键弱点。



## **37. Structure Disruption: Subverting Malicious Diffusion-Based Inpainting via Self-Attention Query Perturbation**

结构破坏：通过自注意查询扰动颠覆基于恶意扩散的修复 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19425v1) [paper-pdf](http://arxiv.org/pdf/2505.19425v1)

**Authors**: Yuhao He, Jinyu Tian, Haiwei Wu, Jianqing Li

**Abstract**: The rapid advancement of diffusion models has enhanced their image inpainting and editing capabilities but also introduced significant societal risks. Adversaries can exploit user images from social media to generate misleading or harmful content. While adversarial perturbations can disrupt inpainting, global perturbation-based methods fail in mask-guided editing tasks due to spatial constraints. To address these challenges, we propose Structure Disruption Attack (SDA), a powerful protection framework for safeguarding sensitive image regions against inpainting-based editing. Building upon the contour-focused nature of self-attention mechanisms of diffusion models, SDA optimizes perturbations by disrupting queries in self-attention during the initial denoising step to destroy the contour generation process. This targeted interference directly disrupts the structural generation capability of diffusion models, effectively preventing them from producing coherent images. We validate our motivation through visualization techniques and extensive experiments on public datasets, demonstrating that SDA achieves state-of-the-art (SOTA) protection performance while maintaining strong robustness.

摘要: 扩散模型的快速发展增强了它们的图像修复和编辑能力，但也带来了重大的社会风险。攻击者可以利用来自社交媒体的用户图像生成误导性或有害的内容。虽然对抗性扰动可能会破坏修复，但由于空间限制，基于全局扰动的方法在掩码引导的编辑任务中失败。为了解决这些挑战，我们提出了结构破坏攻击（SDA），一个强大的保护框架，保护敏感的图像区域，防止基于修补的编辑。基于扩散模型的自我注意机制以轮廓为中心的本质，EDA通过在初始去噪步骤期间扰乱自我注意力的查询以破坏轮廓生成过程来优化扰动。这种有针对性的干扰直接破坏了扩散模型的结构生成能力，有效地阻止它们产生连贯的图像。我们通过可视化技术和对公共数据集的广泛实验来验证我们的动机，证明EDA在保持强大的鲁棒性的同时实现了最先进的（SOTA）保护性能。



## **38. Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains**

时间序列基础模型是否已准备好部署？跨领域对抗稳健性的系统研究 cs.LG

Preprint

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19397v1) [paper-pdf](http://arxiv.org/pdf/2505.19397v1)

**Authors**: Jiawen Zhang, Zhenwei Zhang, Shun Zheng, Xumeng Wen, Jia Li, Jiang Bian

**Abstract**: Time Series Foundation Models (TSFMs), which are pretrained on large-scale, cross-domain data and capable of zero-shot forecasting in new scenarios without further training, are increasingly adopted in real-world applications. However, as the zero-shot forecasting paradigm gets popular, a critical yet overlooked question emerges: Are TSFMs robust to adversarial input perturbations? Such perturbations could be exploited in man-in-the-middle attacks or data poisoning. To address this gap, we conduct a systematic investigation into the adversarial robustness of TSFMs. Our results show that even minimal perturbations can induce significant and controllable changes in forecast behaviors, including trend reversal, temporal drift, and amplitude shift, posing serious risks to TSFM-based services. Through experiments on representative TSFMs and multiple datasets, we reveal their consistent vulnerabilities and identify potential architectural designs, such as structural sparsity and multi-task pretraining, that may improve robustness. Our findings offer actionable guidance for designing more resilient forecasting systems and provide a critical assessment of the adversarial robustness of TSFMs.

摘要: 时间序列基础模型（TSFM）在大规模、跨域数据上进行预训练，并且能够在新场景中进行零触发预测，而无需进一步训练，越来越多地被应用于现实世界的应用程序中。然而，随着零镜头预测范式的流行，一个关键但被忽视的问题出现了：TSFM对对抗性输入扰动是否稳健？此类扰动可能会被利用在中间人攻击或数据中毒中。为了解决这一差距，我们对TSFM的对抗稳健性进行了系统调查。我们的结果表明，即使是最小的扰动也会导致预测行为发生显着且可控的变化，包括趋势逆转、时间漂移和幅度漂移，从而对基于TSFM的服务构成严重风险。通过对代表性TSFM和多个数据集的实验，我们揭示了它们的一致漏洞，并识别了可能提高稳健性的潜在架构设计，例如结构稀疏性和多任务预训练。我们的研究结果为设计更具弹性的预测系统提供了可操作的指导，并对TSFM的对抗稳健性进行了严格评估。



## **39. RADEP: A Resilient Adaptive Defense Framework Against Model Extraction Attacks**

RADPP：针对模型提取攻击的弹性自适应防御框架 cs.CR

Presented at the IEEE International Wireless Communications and  Mobile Computing Conference (IWCMC) 2025

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19364v1) [paper-pdf](http://arxiv.org/pdf/2505.19364v1)

**Authors**: Amit Chakraborty, Sayyed Farid Ahamed, Sandip Roy, Soumya Banerjee, Kevin Choi, Abdul Rahman, Alison Hu, Edward Bowen, Sachin Shetty

**Abstract**: Machine Learning as a Service (MLaaS) enables users to leverage powerful machine learning models through cloud-based APIs, offering scalability and ease of deployment. However, these services are vulnerable to model extraction attacks, where adversaries repeatedly query the application programming interface (API) to reconstruct a functionally similar model, compromising intellectual property and security. Despite various defense strategies being proposed, many suffer from high computational costs, limited adaptability to evolving attack techniques, and a reduction in performance for legitimate users. In this paper, we introduce a Resilient Adaptive Defense Framework for Model Extraction Attack Protection (RADEP), a multifaceted defense framework designed to counteract model extraction attacks through a multi-layered security approach. RADEP employs progressive adversarial training to enhance model resilience against extraction attempts. Malicious query detection is achieved through a combination of uncertainty quantification and behavioral pattern analysis, effectively identifying adversarial queries. Furthermore, we develop an adaptive response mechanism that dynamically modifies query outputs based on their suspicion scores, reducing the utility of stolen models. Finally, ownership verification is enforced through embedded watermarking and backdoor triggers, enabling reliable identification of unauthorized model use. Experimental evaluations demonstrate that RADEP significantly reduces extraction success rates while maintaining high detection accuracy with minimal impact on legitimate queries. Extensive experiments show that RADEP effectively defends against model extraction attacks and remains resilient even against adaptive adversaries, making it a reliable security framework for MLaaS models.

摘要: 机器学习即服务（MLaSaaS）使用户能够通过基于云的API利用强大的机器学习模型，提供可扩展性和易于部署的功能。然而，这些服务很容易受到模型提取攻击，对手会反复查询应用程序编程接口（API）以重建功能相似的模型，从而损害知识产权和安全性。尽管提出了各种防御策略，但许多策略都面临着计算成本高、对不断发展的攻击技术的适应性有限以及合法用户的性能下降的问题。本文中，我们介绍了一个用于模型提取攻击保护（RADPP）的弹性自适应防御框架，这是一个多方面的防御框架，旨在通过多层安全方法对抗模型提取攻击。RADPP采用渐进式对抗训练来增强模型针对提取尝试的弹性。恶意查询检测是通过结合不确定性量化和行为模式分析来实现的，有效识别对抗性查询。此外，我们开发了一种自适应响应机制，该机制根据怀疑分数动态修改查询输出，从而减少被盗模型的效用。最后，通过嵌入式水印和后门触发器强制所有权验证，从而能够可靠地识别未经授权的模型使用。实验评估表明，RADPP显着降低了提取成功率，同时保持了高检测准确性，对合法查询的影响最小。大量实验表明，RADPP可以有效地抵御模型提取攻击，并且即使对自适应对手也保持弹性，使其成为MLaSaaS模型的可靠安全框架。



## **40. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

弯曲动态黑匣子攻击：通过动态弯曲估计重新审视对抗鲁棒性 cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19194v1) [paper-pdf](http://arxiv.org/pdf/2505.19194v1)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.

摘要: 对抗性攻击揭示了深度学习模型的脆弱性。大约十年来，人们提出了无数的攻击和防御方法，从而产生了鲁棒化分类器并更好地理解模型。在这些方法中，基于弯曲的方法引起了人们的关注，因为人们认为高弯曲可能会产生粗略的决策边界。然而，最常用的\textit{currency}是模型内的损失函数、分数或其他参数的弯曲，而不是决策边界弯曲，因为前者可以相对容易地使用二阶求导形成。在本文中，我们提出了一种新的查询高效方法--动态弯曲估计（VCE），来估计黑匣子环境下的决策边界弯曲。我们的方法基于CGBA，这是一种黑匣子对抗攻击。通过对广泛的分类器执行VCE，我们从统计上发现了决策边界弯曲和对抗鲁棒性之间的联系。我们还提出了一种新的攻击方法：弯曲动态黑匣子攻击（CDBA），使用动态估计的弯曲来提高性能。



## **41. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

具有事后感知校准的潜在空间对抗训练，用于保护大型语言模型免受越狱攻击 cs.CR

Under Review

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2501.10639v2) [paper-pdf](http://arxiv.org/pdf/2501.10639v2)

**Authors**: Xin Yi, Yue Li, dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks.

摘要: 确保安全一致已成为大型语言模型（LLM）的关键要求，特别是考虑到它们在现实世界应用程序中的广泛部署。然而，LLM仍然容易受到越狱攻击，这些攻击利用系统漏洞绕过安全措施并产生有害输出。尽管已经提出了许多基于对抗训练的防御机制，但一个持续的挑战在于过度拒绝行为的加剧，这损害了该模型的整体实用性。为了应对这些挑战，我们提出了一种具有事后感知校准的潜在空间对抗训练（LAPC）框架。在对抗训练阶段，LAPC比较潜在空间中的有害和无害指令，并提取安全关键维度来构建拒绝特征攻击，精确模拟需要对抗缓解的不可知越狱攻击类型。在推理阶段，采用嵌入级校准机制以最小的计算负担减轻过度拒绝行为。实验结果表明，与五种越狱攻击的各种防御方法相比，LAPC框架在安全性和实用性之间实现了更好的平衡。此外，我们的分析强调了从潜在空间中提取安全关键维度以构建稳健拒绝特征攻击的有效性。



## **42. PII-Scope: A Comprehensive Study on Training Data PII Extraction Attacks in LLMs**

PII-Scope：LLM中训练数据PRI提取攻击的综合研究 cs.CL

Additional results with Pythia6.9B; Additional results with Phone  number PII;

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.06704v2) [paper-pdf](http://arxiv.org/pdf/2410.06704v2)

**Authors**: Krishna Kanth Nakka, Ahmed Frikha, Ricardo Mendes, Xue Jiang, Xuebing Zhou

**Abstract**: In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.

摘要: 在这项工作中，我们引入了PII-Scope，这是一个全面的基准，旨在评估针对不同威胁环境下LLM的PRI提取攻击的最新方法。我们的研究通过揭示几个超参数（例如，演示选择）对其有效性至关重要。基于这一理解，我们将研究扩展到更现实的攻击场景，探索采用高级对抗策略的PRI攻击，包括重复和多样化的查询，并利用迭代学习来持续的PRI提取。通过广泛的实验，我们的结果揭示了现有单查询攻击中对PRI泄漏的显着低估。事实上，我们表明，由于复杂的对抗能力和有限的查询预算，当针对预训练模型时，PIP提取率可以增加多达五倍。此外，我们评估了微调模型上的PRI泄漏，表明它们比预训练模型更容易受到泄漏的影响。总体而言，我们的工作为现实威胁场景中的PRI提取攻击建立了严格的经验基准，并为制定有效的缓解策略提供了坚实的基础。



## **43. Understanding the Robustness of Graph Neural Networks against Adversarial Attacks**

了解图神经网络对抗对抗攻击的鲁棒性 cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2406.13920v2) [paper-pdf](http://arxiv.org/pdf/2406.13920v2)

**Authors**: Tao Wu, Canyixing Cui, Xingping Xian, Shaojie Qiao, Chao Wang, Lin Yuan, Shui Yu

**Abstract**: Recent studies have shown that graph neural networks (GNNs) are vulnerable to adversarial attacks, posing significant challenges to their deployment in safety-critical scenarios. This vulnerability has spurred a growing focus on designing robust GNNs. Despite this interest, current advancements have predominantly relied on empirical trial and error, resulting in a limited understanding of the robustness of GNNs against adversarial attacks. To address this issue, we conduct the first large-scale systematic study on the adversarial robustness of GNNs by considering the patterns of input graphs, the architecture of GNNs, and their model capacity, along with discussions on sensitive neurons and adversarial transferability. This work proposes a comprehensive empirical framework for analyzing the adversarial robustness of GNNs. To support the analysis of adversarial robustness in GNNs, we introduce two evaluation metrics: the confidence-based decision surface and the accuracy-based adversarial transferability rate. Through experimental analysis, we derive 11 actionable guidelines for designing robust GNNs, enabling model developers to gain deeper insights. The code of this study is available at https://github.com/star4455/GraphRE.

摘要: 最近的研究表明，图神经网络（GNN）容易受到对抗攻击，这对其在安全关键场景中的部署构成了重大挑战。该漏洞促使人们越来越关注设计稳健的GNN。尽管存在这种兴趣，但当前的进步主要依赖于经验性的试错，导致人们对GNN对抗攻击的稳健性的了解有限。为了解决这个问题，我们通过考虑输入图的模式、GNN的架构及其模型容量，以及对敏感神经元和对抗可移植性的讨论，对GNN的对抗鲁棒性进行了首次大规模系统研究。这项工作提出了一个全面的经验框架来分析GNN的对抗稳健性。为了支持GNN中对抗鲁棒性的分析，我们引入了两个评估指标：基于置信度的决策面和基于准确度的对抗可转移率。通过实验分析，我们得出了11个可操作的指导方针，用于设计稳健的GNN，使模型开发人员能够获得更深入的见解。本研究的代码可在https://github.com/star4455/GraphRE上获取。



## **44. A quantitative notion of economic security for smart contract compositions**

智能合约组合的经济安全量化概念 cs.CR

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19006v1) [paper-pdf](http://arxiv.org/pdf/2505.19006v1)

**Authors**: Emily Priyadarshini, Massimo Bartoletti

**Abstract**: Decentralized applications are often composed of multiple interconnected smart contracts. This is especially evident in DeFi, where protocols are heavily intertwined and rely on a variety of basic building blocks such as tokens, decentralized exchanges and lending protocols. A crucial security challenge in this setting arises when adversaries target individual components to cause systemic economic losses. Existing security notions focus on determining the existence of these attacks, but fail to quantify the effect of manipulating individual components on the overall economic security of the system. In this paper, we introduce a quantitative security notion that measures how an attack on a single component can amplify economic losses of the overall system. We study the fundamental properties of this notion and apply it to assess the security of key compositions. In particular, we analyse under-collateralized loan attacks in systems made of lending protocols and decentralized exchanges.

摘要: 去中心化的应用程序通常由多个相互连接的智能合约组成。这在DeFi中尤为明显，其中协议相互交织，并依赖于各种基本构建块，如令牌，去中心化交易所和借贷协议。在这种情况下，当对手瞄准单个组件造成系统性经济损失时，就会出现关键的安全挑战。现有的安全概念侧重于确定这些攻击的存在，但未能量化操纵单个组件对系统整体经济安全的影响。在本文中，我们引入了一种量化安全概念，该概念衡量对单个组件的攻击如何放大整个系统的经济损失。我们研究这个概念的基本属性，并将其应用于评估关键成分的安全性。特别是，我们分析了由贷款协议和去中心化交易所组成的系统中的抵押不足贷款攻击。



## **45. A theoretical basis for MEV**

MEV的理论基础 cs.CR

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2302.02154v5) [paper-pdf](http://arxiv.org/pdf/2302.02154v5)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Maximal Extractable Value (MEV) refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream DeFi protocols are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the increasing real-world impact of these attacks, their theoretical foundations remain insufficiently established. We propose a formal theory of MEV, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against MEV attacks.

摘要: 最大可提取价值（MEV）指的是对公共区块链的广泛一类经济攻击，其中有权重新排序、删除或插入区块交易的对手可以从智能合约中“提取”价值。实证研究表明，主流DeFi协议成为这些攻击的大规模目标，对其用户和区块链网络产生了不利影响。尽管这些攻击对现实世界的影响越来越大，但它们的理论基础仍然不够建立。我们基于区块链和智能合约的一般抽象模型，提出了MEV的形式化理论。我们的理论是针对MEV攻击的安全性证明的基础。



## **46. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

24 pages, 5 tables, 9 figures, Camera-ready version accepted to FSE  2025

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.22663v4) [paper-pdf](http://arxiv.org/pdf/2410.22663v4)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.

摘要: 用于文本分类的机器学习（ML）已广泛应用于各个领域。这些应用程序可能会显着影响道德、经济和人类行为，从而引发人们对信任ML决策的严重担忧。研究表明，传统指标不足以建立人类对ML模型的信任。这些模型经常学习虚假相关性并基于它们进行预测。在现实世界中，他们的表现可能会显着恶化。为了避免这种情况，常见的做法是根据数据中的有效模式测试预测是否合理。与此同时，还引入了一个称为可信度Oracle问题的挑战。由于缺乏自动可信度预言，评估需要对解释方法披露的决策过程进行手动验证。然而，这耗时、容易出错且不可扩展。   我们提出了TOKI，这是第一种文本分类器的自动可信Oracle生成方法。TOKI自动检查对预测贡献最大的单词是否与预测的类别在语义上相关。具体来说，我们利用ML解释来提取影响决策的单词，并基于单词嵌入来测量它们与类的语义相关性。我们还引入了一种新颖的对抗攻击方法，该方法针对TOKI识别的可信度漏洞。为了评估它们与人类判断的一致性，我们进行了实验。我们将TOKI与仅基于模型置信度和TOKI引导的对抗攻击方法与A2 T（一种SOTA对抗攻击方法）进行比较。结果表明，依赖预测不确定性无法有效区分可信和不可信的预测，TOKI的准确性比原始基线高出142%，TOKI引导的攻击方法比A2 T更有效，干扰更少。



## **47. GhostPrompt: Jailbreaking Text-to-image Generative Models based on Dynamic Optimization**

GhostPrompt：基于动态优化的越狱文本到图像生成模型 cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18979v1) [paper-pdf](http://arxiv.org/pdf/2505.18979v1)

**Authors**: Zixuan Chen, Hao Lin, Ke Xu, Xinghao Jiang, Tanfeng Sun

**Abstract**: Text-to-image (T2I) generation models can inadvertently produce not-safe-for-work (NSFW) content, prompting the integration of text and image safety filters. Recent advances employ large language models (LLMs) for semantic-level detection, rendering traditional token-level perturbation attacks largely ineffective. However, our evaluation shows that existing jailbreak methods are ineffective against these modern filters. We introduce GhostPrompt, the first automated jailbreak framework that combines dynamic prompt optimization with multimodal feedback. It consists of two key components: (i) Dynamic Optimization, an iterative process that guides a large language model (LLM) using feedback from text safety filters and CLIP similarity scores to generate semantically aligned adversarial prompts; and (ii) Adaptive Safety Indicator Injection, which formulates the injection of benign visual cues as a reinforcement learning problem to bypass image-level filters. GhostPrompt achieves state-of-the-art performance, increasing the ShieldLM-7B bypass rate from 12.5\% (Sneakyprompt) to 99.0\%, improving CLIP score from 0.2637 to 0.2762, and reducing the time cost by $4.2 \times$. Moreover, it generalizes to unseen filters including GPT-4.1 and successfully jailbreaks DALLE 3 to generate NSFW images in our evaluation, revealing systemic vulnerabilities in current multimodal defenses. To support further research on AI safety and red-teaming, we will release code and adversarial prompts under a controlled-access protocol.

摘要: 文本到图像（T2 I）生成模型可能会无意中产生不安全工作（NSFW）内容，从而促使文本和图像安全过滤器的集成。最近的进展使用大型语言模型（LLM）进行语义级检测，使传统的符号级扰动攻击基本上无效。然而，我们的评估表明，现有的越狱方法对这些现代过滤器无效。我们引入GhostPrompt，这是第一个自动越狱框架，将动态提示优化与多模式反馈相结合。它由两个关键组件组成：（i）动态优化，这是一个迭代过程，使用来自文本安全过滤器的反馈和CLIP相似性分数来指导大型语言模型（LLM）生成语义对齐的对抗提示;和（ii）自适应安全指标注入，它将良性视觉线索的注入制定为强化学习问题，以绕过图像级过滤器。GhostPrompt实现了最先进的性能，将ShieldLM-7 B旁路率从12.5%（Sneakypromit）提高到99.0%，将CLIP评分从0.2637提高到0.2762，并将时间成本减少4.2美元。此外，它还推广到了包括GPT-4.1在内的不可见过滤器，并在我们的评估中成功越狱DALLE 3以生成NSFW图像，揭示了当前多模式防御中的系统性漏洞。为了支持对人工智能安全和红色团队的进一步研究，我们将在受控访问协议下发布代码和对抗提示。



## **48. Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services**

预训练的编码器推理：揭示下游机器学习服务中的上游编码器 cs.LG

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2408.02814v2) [paper-pdf](http://arxiv.org/pdf/2408.02814v2)

**Authors**: Shaopeng Fu, Xuexue Sun, Ke Qing, Tianhang Zheng, Di Wang

**Abstract**: Pre-trained encoders available online have been widely adopted to build downstream machine learning (ML) services, but various attacks against these encoders also post security and privacy threats toward such a downstream ML service paradigm. We unveil a new vulnerability: the Pre-trained Encoder Inference (PEI) attack, which can extract sensitive encoder information from a targeted downstream ML service that can then be used to promote other ML attacks against the targeted service. By only providing API accesses to a targeted downstream service and a set of candidate encoders, the PEI attack can successfully infer which encoder is secretly used by the targeted service based on candidate ones. Compared with existing encoder attacks, which mainly target encoders on the upstream side, the PEI attack can compromise encoders even after they have been deployed and hidden in downstream ML services, which makes it a more realistic threat. We empirically verify the effectiveness of the PEI attack on vision encoders. we first conduct PEI attacks against two downstream services (i.e., image classification and multimodal generation), and then show how PEI attacks can facilitate other ML attacks (i.e., model stealing attacks vs. image classification models and adversarial attacks vs. multimodal generative models). Our results call for new security and privacy considerations when deploying encoders in downstream services. The code is available at https://github.com/fshp971/encoder-inference.

摘要: 在线预训练编码器已被广泛采用来构建下游机器学习（ML）服务，但针对这些编码器的各种攻击也对此类下游ML服务范式构成了安全和隐私威胁。我们揭示了一个新的漏洞：预训练的编码器推断（PHI）攻击，可以从目标下游ML服务中提取敏感的编码器信息，然后可以使用这些信息来促进针对目标服务的其他ML攻击。通过仅提供对目标下游服务和一组候选编码器的API访问，Pe攻击可以根据候选编码器成功推断目标服务秘密使用哪个编码器。与主要针对上游编码器的现有编码器攻击相比，即使编码器已被部署并隐藏在下游ML服务中，PRI攻击也可以危及编码器，这使得其成为一个更现实的威胁。我们通过经验验证了Pe对视觉编码器攻击的有效性。我们首先对两个下游服务进行PPE攻击（即，图像分类和多模式生成），然后展示EDI攻击如何促进其他ML攻击（即，模型窃取攻击与图像分类模型以及对抗攻击与多模式生成模型）。我们的结果要求在下游服务中部署编码器时考虑新的安全和隐私考虑因素。该代码可在https://github.com/fshp971/encoder-inference上获取。



## **49. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全性问题综述 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18889v1) [paper-pdf](http://arxiv.org/pdf/2505.18889v1)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 (and its recent iterations like GPT-4o and the GPT-4.1 series), Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks (including input perturbations and data poisoning), misuse by malicious actors (e.g., for disinformation, phishing, and malware generation), and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives (scheming), which may even persist through safety training. We summarize recent academic and industrial studies (2022-2025) that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: 大型语言模型（LLM），例如GPT-4（及其最近的迭代，例如GPT-4 o和GPT-4.1系列）、谷歌的Gemini、Anthropic的Claude 3模型和xAI的Grok，已经引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。在本调查中，我们全面概述了LLM周围新出现的安全问题，将威胁分为即时注入和越狱、对抗性攻击（包括输入扰动和数据中毒）、恶意行为者的滥用（例如，虚假信息、网络钓鱼和恶意软件生成），以及自主LLM代理固有的令人担忧的风险。最近人们对后者给予了极大的关注，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标（阴谋）的潜力，甚至可能通过安全培训持续存在。我们总结了最近的学术和工业研究（2022-2025年），这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **50. Audio Jailbreak Attacks: Exposing Vulnerabilities in SpeechGPT in a White-Box Framework**

音频越狱攻击：在白盒框架中暴露SpeechGPT中的漏洞 cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18864v1) [paper-pdf](http://arxiv.org/pdf/2505.18864v1)

**Authors**: Binhao Ma, Hanqing Guo, Zhengping Jay Luo, Rui Duan

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced the naturalness and flexibility of human computer interaction by enabling seamless understanding across text, vision, and audio modalities. Among these, voice enabled models such as SpeechGPT have demonstrated considerable improvements in usability, offering expressive, and emotionally responsive interactions that foster deeper connections in real world communication scenarios. However, the use of voice introduces new security risks, as attackers can exploit the unique characteristics of spoken language, such as timing, pronunciation variability, and speech to text translation, to craft inputs that bypass defenses in ways not seen in text-based systems. Despite substantial research on text based jailbreaks, the voice modality remains largely underexplored in terms of both attack strategies and defense mechanisms. In this work, we present an adversarial attack targeting the speech input of aligned MLLMs in a white box scenario. Specifically, we introduce a novel token level attack that leverages access to the model's speech tokenization to generate adversarial token sequences. These sequences are then synthesized into audio prompts, which effectively bypass alignment safeguards and to induce prohibited outputs. Evaluated on SpeechGPT, our approach achieves up to 89 percent attack success rate across multiple restricted tasks, significantly outperforming existing voice based jailbreak methods. Our findings shed light on the vulnerabilities of voice-enabled multimodal systems and to help guide the development of more robust next-generation MLLMs.

摘要: 多模式大型语言模型（MLLM）的最新进展通过实现文本、视觉和音频模式的无缝理解，显着增强了人机交互的自然性和灵活性。其中，SpeechGPT等语音支持模型在可用性方面表现出了相当大的改进，提供了富有表达力和情感响应的交互，从而在现实世界通信场景中促进了更深层次的联系。然而，语音的使用会带来新的安全风险，因为攻击者可以利用口语的独特特征，例如时间、发音变异性以及语音到文本的翻译，以基于文本的系统中所没有的方式制作绕过防御的输入。尽管对基于文本的越狱进行了大量研究，但语音模式在攻击策略和防御机制方面仍然很大程度上没有得到充分的研究。在这项工作中，我们提出了一种针对白盒场景中对齐MLLM的语音输入的对抗攻击。具体来说，我们引入了一种新型的令牌级攻击，该攻击利用对模型的语音令牌化的访问来生成对抗性令牌序列。然后，这些序列被合成为音频提示，从而有效地绕过对齐保护措施并引发被禁止的输出。经过SpeechGPT评估，我们的方法在多个受限任务中实现了高达89%的攻击成功率，显着优于现有的基于语音的越狱方法。我们的研究结果揭示了语音多模式系统的漏洞，并帮助指导更强大的下一代MLLM的开发。



