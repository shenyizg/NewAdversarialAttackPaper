# Latest Adversarial Attack Papers
**update at 2025-10-24 10:13:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Tex-ViT: A Generalizable, Robust, Texture-based dual-branch cross-attention deepfake detector**

Tex-ViT：一种可推广、稳健、基于纹理的双分支交叉注意深度伪造检测器 cs.CV

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2408.16892v2) [paper-pdf](http://arxiv.org/pdf/2408.16892v2)

**Authors**: Deepak Dagar, Dinesh Kumar Vishwakarma

**Abstract**: Deepfakes, which employ GAN to produce highly realistic facial modification, are widely regarded as the prevailing method. Traditional CNN have been able to identify bogus media, but they struggle to perform well on different datasets and are vulnerable to adversarial attacks due to their lack of robustness. Vision transformers have demonstrated potential in the realm of image classification problems, but they require enough training data. Motivated by these limitations, this publication introduces Tex-ViT (Texture-Vision Transformer), which enhances CNN features by combining ResNet with a vision transformer. The model combines traditional ResNet features with a texture module that operates in parallel on sections of ResNet before each down-sampling operation. The texture module then serves as an input to the dual branch of the cross-attention vision transformer. It specifically focuses on improving the global texture module, which extracts feature map correlation. Empirical analysis reveals that fake images exhibit smooth textures that do not remain consistent over long distances in manipulations. Experiments were performed on different categories of FF++, such as DF, f2f, FS, and NT, together with other types of GAN datasets in cross-domain scenarios. Furthermore, experiments also conducted on FF++, DFDCPreview, and Celeb-DF dataset underwent several post-processing situations, such as blurring, compression, and noise. The model surpassed the most advanced models in terms of generalization, achieving a 98% accuracy in cross-domain scenarios. This demonstrates its ability to learn the shared distinguishing textural characteristics in the manipulated samples. These experiments provide evidence that the proposed model is capable of being applied to various situations and is resistant to many post-processing procedures.

摘要: Deepfakes使用GAN来制作高度逼真的面部修饰，被广泛认为是流行的方法。传统的CNN已经能够识别虚假媒体，但它们很难在不同的数据集上表现良好，并且由于缺乏稳健性而容易受到对抗攻击。视觉转换器在图像分类问题领域已经展现出潜力，但它们需要足够的训练数据。受这些限制的启发，本出版物引入了Tex-ViT（纹理视觉Transformer），它通过将ResNet与视觉Transformer相结合来增强CNN功能。该模型将传统的ResNet特征与纹理模块相结合，该纹理模块在每次下采样操作之前对ResNet的部分进行并行操作。然后，纹理模块用作交叉注意视觉Transformer的双分支的输入。它特别注重改进提取特征地图相关性的全局纹理模块。经验分析表明，假图像表现出平滑的纹理，在操纵中在长距离内不会保持一致。在跨域场景中对不同类别的FF++（例如DF、f2 f、FS和NT）以及其他类型的GAN数据集进行了实验。此外，还对FF++、DFDCPreview和Celeb-DF数据集进行了实验，经历了多种后处理情况，例如模糊、压缩和噪音。该模型在概括性方面超越了最先进的模型，在跨领域场景中实现了98%的准确率。这证明了它能够学习处理样本中共同的区分纹理特征。这些实验提供了证据，表明所提出的模型能够应用于各种情况，并且能够抵抗许多后处理过程。



## **2. AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN**

AdaNOS：通过SDK中的深度对抗强化学习的自适应NOS攻击 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20566v1) [paper-pdf](http://arxiv.org/pdf/2510.20566v1)

**Authors**: Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe

**Abstract**: Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors.

摘要: 现有的防御机制在缓解基于规则的拒绝服务（DPS）攻击、利用预定义的签名和静态启发法来识别和阻止恶意流量方面表现出显着的有效性。然而，人工智能驱动技术的出现给SDK安全带来了新的挑战，可能会损害现有防御机制的有效性。本文中，我们介绍了~ AdaNOS，这是一种自适应攻击模型，可以通过对抗强化学习（RL）扰乱网络操作，同时逃避现有基于DoS的检测器的检测。具体来说，AdaNOS将该问题建模为攻击者和检测器之间的竞争游戏，攻击者的目标是在不被检测到的情况下阻止网络流量，而检测器的目标是识别恶意流量。AdaNOS可以根据来自dn和检测器的反馈动态调整其攻击策略来解决这个问题。此外，由于认识到攻击者通常比防御者拥有的信息少，AdaNOS类攻击将其定义为部分观察的马尔科夫决策过程（POMDP），攻击者只能访问攻击者和受害者节点之间的延迟信息。我们通过一个新颖的互惠学习模块来解决这一挑战，其中观察有限的学生代理通过向教师代理学习来提高其性能，教师代理在dn环境中具有完整的观察能力。AdaNOS代表了RL的第一个应用程序来开发类似于DoS的攻击序列，能够自适应地规避基于机器学习和基于规则的类似于DoS的攻击检测器。



## **3. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出非凡的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个关键问题出现了：当推理与危害交织在一起时，LRM是否会在推理模式中变得更容易越狱？为了研究这一点，我们引入了HauntAttack，这是一种新颖的通用黑匣子对抗攻击框架，它系统地将有害指令嵌入到推理问题中。具体来说，我们用有害指令修改现有问题中的关键推理条件，从而构建一条推理路径，引导模型逐步走向不安全的输出。我们对11种LRM进行了HauntAttack评估，观察到平均攻击成功率为70%，比之前最强的基线实现了高达12个百分点的绝对改进。我们的进一步分析表明，即使是先进的安全性一致的模型仍然极易受到基于推理的攻击，这为未来模型开发中平衡推理能力和安全性的紧迫挑战提供了见解。



## **4. Distributional Adversarial Attacks and Training in Deep Hedging**

分布式对抗攻击和深度对冲培训 math.OC

Camera-ready version (accepted at NeurIPS 2025  https://neurips.cc/virtual/2025/poster/115434)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2508.14757v2) [paper-pdf](http://arxiv.org/pdf/2508.14757v2)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Additional results indicate that the robust strategies maintain reliable performance on real market data and remain effective during periods of market change. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.

摘要: 本文通过利用对抗攻击的概念，研究了分布变化下经典深度对冲策略的稳健性。我们首先证明，标准的深度对冲模型极易受到输入分布中的微小扰动的影响，从而导致性能显着下降。受此启发，我们提出了一个对抗性训练框架，旨在提高深度对冲策略的稳健性。我们的方法将逐点对抗攻击扩展到分布式环境，并在Wasserstein球上引入了对抗优化问题的计算易于处理的重新公式。这使得能够有效地训练对分布扰动有弹性的对冲策略。通过广泛的数值实验，我们表明，在样本外性能和对模型错误指定的弹性方面，经过对抗训练的深度对冲策略始终优于经典对冲策略。其他结果表明，稳健的策略可以在真实市场数据上保持可靠的表现，并在市场变化期间保持有效。我们的研究结果为现实市场不确定性下的稳健深度对冲建立了一个实用有效的框架。



## **5. GUIDE: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models**

指南：通过去噪模型增强联邦学习中的梯度倒置攻击 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.17621v2) [paper-pdf](http://arxiv.org/pdf/2510.17621v2)

**Authors**: Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella, Mario Vento

**Abstract**: Federated Learning (FL) enables collaborative training of Machine Learning (ML) models across multiple clients while preserving their privacy. Rather than sharing raw data, federated clients transmit locally computed updates to train the global model. Although this paradigm should provide stronger privacy guarantees than centralized ML, client updates remain vulnerable to privacy leakage. Adversaries can exploit them to infer sensitive properties about the training data or even to reconstruct the original inputs via Gradient Inversion Attacks (GIAs). Under the honest-butcurious threat model, GIAs attempt to reconstruct training data by reversing intermediate updates using optimizationbased techniques. We observe that these approaches usually reconstruct noisy approximations of the original inputs, whose quality can be enhanced with specialized denoising models. This paper presents Gradient Update Inversion with DEnoising (GUIDE), a novel methodology that leverages diffusion models as denoising tools to improve image reconstruction attacks in FL. GUIDE can be integrated into any GIAs that exploits surrogate datasets, a widely adopted assumption in GIAs literature. We comprehensively evaluate our approach in two attack scenarios that use different FL algorithms, models, and datasets. Our results demonstrate that GUIDE integrates seamlessly with two state-ofthe- art GIAs, substantially improving reconstruction quality across multiple metrics. Specifically, GUIDE achieves up to 46% higher perceptual similarity, as measured by the DreamSim metric.

摘要: 联合学习（FL）支持跨多个客户端对机器学习（ML）模型进行协作训练，同时保护他们的隐私。联邦客户端不是共享原始数据，而是传输本地计算的更新来训练全球模型。尽管这种范式应该比集中式ML提供更强的隐私保证，但客户端更新仍然容易受到隐私泄露的影响。对手可以利用它们来推断训练数据的敏感属性，甚至通过梯度倒置攻击（GIA）重建原始输入。在诚实但好奇的威胁模型下，GIA试图通过使用基于优化的技术逆转中间更新来重建训练数据。我们观察到这些方法通常重建原始输入的有噪逼近，其质量可以通过专门的去噪模型来增强。本文介绍了带去噪的梯度更新倒置（GUIDE），这是一种新颖的方法，利用扩散模型作为去噪工具来改善FL中的图像重建攻击。GUIDE可以集成到任何利用代理数据集的GIA中，这是GIA文献中广泛采用的假设。我们在使用不同FL算法、模型和数据集的两种攻击场景中全面评估了我们的方法。我们的结果表明，GUIDE与两个最先进的GIA无缝集成，大幅提高了多个指标的重建质量。具体来说，通过DreamSim指标衡量，GUIDE的感知相似性提高了46%。



## **6. GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?**

GhostEEI-Bench：移动代理对动态设备上环境中的环境注入有弹性吗？ cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20333v1) [paper-pdf](http://arxiv.org/pdf/2510.20333v1)

**Authors**: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.

摘要: 视觉语言模型（VLM）越来越多地被部署为自治代理来导航移动图形用户界面（GUIs）。在动态的设备上生态系统（包括通知、弹出窗口和应用程序间交互）中运行，使它们面临一种独特且未充分探索的威胁载体：环境注入。与操纵文本指令的基于预算的攻击不同，环境注入通过将对抗性UI元素（例如，欺骗性覆盖或欺骗通知）直接插入到图形用户界面中来破坏代理的视觉感知。这绕过了文本保护措施，并可能会导致执行脱轨，导致隐私泄露、财务损失或不可逆转的设备损害。为了系统性地评估这种威胁，我们引入了GhostEI-Bench，这是第一个用于评估动态可执行环境中环境注入攻击下的移动代理的基准。GhostEEI-Bench超越了基于静态图像的评估，将对抗事件注入到完全运行的Android模拟器内的现实应用程序工作流程中，并评估关键风险场景中的性能。我们进一步提出了一种判断LLM协议，该协议通过审查代理的动作轨迹以及相应的屏幕截图序列来进行细粒度的失败分析，从而确定感知、识别或推理中的失败。对最先进代理的全面实验揭示了对欺骗性环境线索的明显脆弱性：当前的模型系统性地无法感知和推理被操纵的UI。GhostEEI-Bench提供了一个量化和缓解这种新出现的威胁的框架，为更强大、更安全的具体化代理铺平了道路。



## **7. Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses**

增强深度强化学习的安全性：对抗性攻击和防御的全面调查 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20314v1) [paper-pdf](http://arxiv.org/pdf/2510.20314v1)

**Authors**: Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun

**Abstract**: With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers.

摘要: 随着深度强化学习（DRL）技术在自动驾驶、智能制造、智能医疗等复杂领域的广泛应用，如何提高其在动态多变环境下的安全性和鲁棒性已成为当前研究的核心问题。特别是面对对抗攻击时，DRL可能会出现严重的性能下降，甚至做出潜在危险的决策，因此确保其在安全敏感场景中的稳定性至关重要。本文首先介绍了DRL的基本框架，并分析了复杂且不断变化的环境中面临的主要安全挑战。此外，本文提出了基于扰动类型和攻击目标的对抗性攻击分类框架，并详细回顾了针对DRL的主流对抗性攻击方法，包括扰动状态空间、动作空间、奖励函数和模型空间等各种攻击方法。为了有效对抗攻击，本文系统总结了当前各种鲁棒性训练策略，包括对抗性训练、竞争性训练、鲁棒学习、对抗性检测、防御提炼等相关防御技术，并讨论了这些方法在提高DRL鲁棒性方面的优点和缺点。最后，本文展望了对抗环境下DRL的未来研究方向，强调了在提高概括性、降低计算复杂性、增强可扩展性和可解释性方面的研究需求，旨在为研究人员提供有价值的参考和方向。



## **8. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

针对表格数据设计不可感知的Manifold对抗攻击 cs.LG

39 pages

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2507.10998v2) [paper-pdf](http://arxiv.org/pdf/2507.10998v2)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present unique challenges due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions. To address this, we propose a latent-space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate statistically consistent adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We introduce In-Distribution Success Rate (IDSR) to jointly evaluate attack effectiveness and distributional alignment. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches, achieving substantially lower outlier rates and higher IDSR across six datasets and three model architectures. Our comprehensive analyses of hyperparameter sensitivity, sparsity control, and generative architecture demonstrate that the effectiveness of VAE-based attacks depends strongly on reconstruction quality and the availability of sufficient training data. When these conditions are met, the proposed framework achieves superior practical utility and stability compared with input-space methods. This work underscores the importance of maintaining on-manifold perturbations for generating realistic and robust adversarial examples in tabular domains.

摘要: 由于混合类别和数字特征的多样性，对表格数据的对抗攻击带来了独特的挑战。与像素扰动保持视觉相似性的图像不同，表格数据缺乏直观的相似性指标，因此很难定义难以察觉的修改。此外，传统的基于梯度的方法优先考虑$\ell_p$-norm约束，通常会产生偏离原始数据分布的对抗性示例。为了解决这个问题，我们提出了一种潜空间扰动框架，使用混合输入变分自动编码器（VAE）来生成统计上一致的对抗示例。提出的VAE将类别嵌入和数字特征集成到统一的潜在多管齐中，从而实现保持统计一致性的扰动。我们引入分布式成功率（IDSR）来联合评估攻击有效性和分布式对齐。对六个公开可用数据集和三个模型架构的评估表明，与传统的输入空间攻击和从图像域方法改编的其他基于VAE的方法相比，我们的方法实现了显着更低的异常值率和更一致的性能，实现了显着更低的异常值率和更高的IDSR跨六个数据集和三个模型架构。我们对超参数敏感性、稀疏性控制和生成式架构的全面分析表明，基于VAE的攻击的有效性在很大程度上取决于重建质量和足够训练数据的可用性。当满足这些条件时，与输入空间方法相比，所提出的框架实现了更好的实际实用性和稳定性。这项工作强调了维持管汇上扰动对于在表格域中生成现实且稳健的对抗示例的重要性。



## **9. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

超越文本：通过感性简单的转换对视觉语言和音频模型进行多模式越狱 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.

摘要: 多模式大型语言模型（MLLM）已经取得了显着的进展，但仍然极易受到利用跨模式处理弱点的对抗攻击的影响。我们对针对视觉语言和音频语言模型的多模式越狱进行了系统性研究，表明即使是简单的感知转换也可以可靠地绕过最先进的安全过滤器。我们的评估涵盖了三个高风险安全类别有害内容、CBRN（化学、生物、放射、核）和CTEM（儿童性剥削材料）的1，900个对抗性提示，针对七个前沿模型进行了测试。我们探索了MLLM攻击技术的有效性，包括FigStep-Pro（视觉关键字分解）、智能掩蔽（语义混淆）和音频扰动（Wave-Echo、Wave-Pitch、Wave-Speed）。结果揭示了严重的漏洞：在感知修改的输入下，具有几乎完美的纯文本安全性（0\%ASB）的模型遭受了超过75%的攻击成功率，而FigStep-Pro在Lama-4变体中实现了高达89%的ASB。基于音频的攻击进一步揭示了提供商特定的弱点，即使是基本的模式传输也会产生25%的技术查询的ASB。这些发现暴露了以文本为中心的对齐和多模式威胁之间的关键差距，表明当前的保障措施未能普遍适用于跨模式攻击。这些攻击的可访问性需要最少的技术专业知识，这表明强大的多模式人工智能安全性需要范式转向更广泛的语义层面推理，以减轻可能的风险。



## **10. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

TRUST：审计大型语言模型推理的去中心化框架 cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.

摘要: 大型语言模型生成复杂的推理链，揭示其决策，但验证这些中间步骤的忠实性和无害性仍然是一个尚未解决的关键问题。现有的审计方法集中、不透明且难以扩展，为在高风险领域部署专有模型带来了巨大风险。我们确定了四个核心挑战：（1）稳健性：集中式审计员是单点失败，容易受到偏见或攻击。(2)可扩展性：推理轨迹太长，无法手动验证。(3)不透明：封闭审计破坏了公众信任。(4)隐私：暴露完整推理可能会导致模型被盗或提炼。我们提出TRUST，这是一个透明的、去中心化的审计框架，通过以下方式克服这些限制：（1）不同审计员之间的共识机制，保证在高达30%的恶意参与者下的正确性。(2)推理痕迹的分层DAB分解，实现可扩展的并行审计。(3)一个区块链分类帐，记录所有验证决定，以供公众问责。(4)保留隐私的分段，仅共享部分推理步骤以保护专有逻辑。我们为TRUST框架的安全性和经济激励提供理论保证。跨多个LLM（GPT-OSS、DeepSeek-r1、Qwen）和推理任务（数学、医学、科学、人文学科）的实验表明，TRUST有效地检测推理缺陷，并在对抗性审计员的情况下保持稳健性。我们的工作开创了去中心化的人工智能审计，为安全且值得信赖的LLM部署提供了实用途径。



## **11. Active Localization of Close-range Adversarial Acoustic Sources for Underwater Data Center Surveillance**

水下数据中心监控中近距离对抗声源的主动定位 eess.SP

12 pages, V1

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20122v1) [paper-pdf](http://arxiv.org/pdf/2510.20122v1)

**Authors**: Adnan Abdullah, David Blow, Sara Rampazzi, Md Jahidul Islam

**Abstract**: Underwater data infrastructures offer natural cooling and enhanced physical security compared to terrestrial facilities, but are susceptible to acoustic injection attacks that can disrupt data integrity and availability. This work presents a comprehensive surveillance framework for localizing and tracking close-range adversarial acoustic sources targeting offshore infrastructures, particularly underwater data centers (UDCs). We propose a heterogeneous receiver configuration comprising a fixed hydrophone mounted on the facility and a mobile hydrophone deployed on a dedicated surveillance robot. While using enough arrays of static hydrophones covering large infrastructures is not feasible in practice, off-the-shelf approaches based on time difference of arrival (TDOA) and frequency difference of arrival (FDOA) filtering fail to generalize for this dynamic configuration. To address this, we formulate a Locus-Conditioned Maximum A-Posteriori (LC-MAP) scheme to generate acoustically informed and geometrically consistent priors, ensuring a physically plausible initial state for a joint TDOA-FDOA filtering. We integrate this into an unscented Kalman filtering (UKF) pipeline, which provides reliable convergence under nonlinearity and measurement noise. Extensive Monte Carlo analyses, Gazebo-based physics simulations, and field trials demonstrate that the proposed framework can reliably estimate the 3D position and velocity of an adversarial acoustic attack source in real time. It achieves sub-meter localization accuracy and over 90% success rates, with convergence times nearly halved compared to baseline methods. Overall, this study establishes a geometry-aware, real-time approach for acoustic threat localization, advancing autonomous surveillance capabilities of underwater infrastructures.

摘要: 与陆地设施相比，水下数据基础设施提供自然冷却和增强的物理安全性，但很容易受到声学注入攻击，从而破坏数据的完整性和可用性。这项工作提供了一个全面的监视框架，用于定位和跟踪针对海上基础设施（特别是水下数据中心（UDC））的近距离对抗性声学源。我们提出了一种异类接收器配置，包括安装在设施上的固定式水下听音器和部署在专用监视机器人上的移动式水下听音器。虽然在实践中使用足够多的静态声纳阵列覆盖大型基础设施是不可行的，但基于到达时间差（TDOE）和到达频率差（FDOA）过滤的现成方法无法普遍适用于这种动态配置。为了解决这个问题，我们制定了一种定位条件最大A后验（LC-MAP）方案来生成声学信息和几何一致的先验，确保联合TDOA-FDOA过滤的物理上合理的初始状态。我们将其集成到无踪迹卡尔曼过滤（UKF）管道中，该管道在非线性和测量噪音下提供可靠的收敛。广泛的蒙特卡洛分析、基于Gazebo的物理模拟和现场试验表明，所提出的框架可以可靠地实时估计对抗性声学攻击源的3D位置和速度。它实现了亚米定位准确度和超过90%的成功率，与基线方法相比，收敛时间几乎减少了一半。总体而言，这项研究建立了一种几何感知的实时声学威胁定位方法，提高水下基础设施的自主监视能力。



## **12. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **13. Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness**

弥合对称性和鲁棒性：论等方差在增强对抗鲁棒性中的作用 cs.LG

Accepted for the proceedings of 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.16171v2) [paper-pdf](http://arxiv.org/pdf/2510.16171v2)

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh, Chaowei Zhang, Xiao Qin, Yang Zhou

**Abstract**: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation- and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.

摘要: 对抗性示例通过利用深度神经网络对不可感知的输入扰动的敏感性来揭示深度神经网络中的关键漏洞。虽然对抗训练仍然是主要的防御策略，但它通常会产生巨大的计算成本，并可能会损害干净数据的准确性。在这项工作中，我们研究了一种对抗鲁棒性的架构方法，通过将组等变卷积（具体来说是旋转和规模等变层）嵌入到标准卷积神经网络（CNN）中。这些层编码对称先验，将模型行为与输入空间中的结构化转换保持一致，从而促进更平滑的决策边界和更大的对抗性攻击弹性。我们提出并评估了两种混合感知架构：在融合之前独立处理标准和等变特征的并行设计，以及顺序应用等变操作的级联设计。从理论上讲，我们证明此类模型可以降低假设空间的复杂性、规范化梯度，并在CIVER（nEtwork稳健性的Cross Lipschitz Extreme Value）框架下产生更严格的认证稳健性界限。从经验上看，我们的模型在FGSM和PVD攻击下一致提高了CIFAR-10、CIFAR-100和CIFAR-10 C的对抗稳健性和概括性，而无需对抗训练。这些发现强调了安全执行架构作为基于数据增强的防御的高效且有原则的替代方案的潜力。



## **14. JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation**

JaiLIP：通过损失引导图像扰动越狱的视觉语言模型 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21401v2) [paper-pdf](http://arxiv.org/pdf/2509.21401v2)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Vision-Language Models (VLMs) have remarkable abilities in generating multimodal reasoning tasks. However, potential misuse or safety alignment concerns of VLMs have increased significantly due to different categories of attack vectors. Among various attack vectors, recent studies have demonstrated that image-based perturbations are particularly effective in generating harmful outputs. In the literature, many existing techniques have been proposed to jailbreak VLMs, leading to unstable performance and visible perturbations. In this study, we propose Jailbreaking with Loss-guided Image Perturbation (JaiLIP), a jailbreaking attack in the image space that minimizes a joint objective combining the mean squared error (MSE) loss between clean and adversarial image with the models harmful-output loss. We evaluate our proposed method on VLMs using standard toxicity metrics from Perspective API and Detoxify. Experimental results demonstrate that our method generates highly effective and imperceptible adversarial images, outperforming existing methods in producing toxicity. Moreover, we have evaluated our method in the transportation domain to demonstrate the attacks practicality beyond toxic text generation in specific domain. Our findings emphasize the practical challenges of image-based jailbreak attacks and the need for efficient defense mechanisms for VLMs.

摘要: 视觉语言模型（VLM）在生成多模式推理任务方面具有非凡的能力。然而，由于攻击载体类型的不同，VLM的潜在误用或安全对齐问题显着增加。在各种攻击载体中，最近的研究表明，基于图像的扰动在产生有害输出方面特别有效。在文献中，许多现有技术都被提出来越狱VLM，导致性能不稳定和可见的扰动。在这项研究中，我们提出了具有损失引导图像扰动的越狱（JaiLIP），这是图像空间中的一种越狱攻击，可最大限度地减少将干净图像和对抗图像之间的均方误差（SSE）损失与模型有害输出损失相结合的联合目标。我们使用Perspective API和Deepfy的标准毒性指标评估我们在VLM上提出的方法。实验结果表明，我们的方法可以生成高效且难以感知的对抗图像，在产生毒性方面优于现有方法。此外，我们在交通领域评估了我们的方法，以证明除了特定领域有毒文本生成之外的攻击的实用性。我们的研究结果强调了基于图像的越狱攻击的实际挑战以及对VLM有效防御机制的需求。



## **15. Sharp Gaussian approximations for Decentralized Federated Learning**

分散式联邦学习的尖锐高斯逼近 stat.ML

Accepted as Spotlight, NeurIPS'25, Main Conference Track

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.08125v2) [paper-pdf](http://arxiv.org/pdf/2505.08125v2)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.

摘要: 联邦学习在隐私敏感的协作环境中获得了吸引力，本地新元正在成为去中心化环境中的关键优化方法。虽然它的收敛性质已得到充分研究，但超越收敛的渐进统计保证仍然有限。在本文中，我们给出了局部SGD的两个广义高斯逼近结果，并探讨了它们的含义。首先，我们证明了最终本地BCD迭代的Berry-Esseen定理，从而实现有效的乘数引导过程。其次，出于鲁棒性考虑，我们为局部SGD的整个轨迹引入了两种不同的时间均匀高斯逼近。时间一致的逼近支持基于高斯引导的测试来检测对抗性攻击。提供了广泛的模拟来支持我们的理论结果。



## **16. QORE : Quantum Secure 5G/B5G Core**

QORE：Quantum安全5G/B5 G核心 cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19982v1) [paper-pdf](http://arxiv.org/pdf/2510.19982v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Rudraksh Rawal, Nitin Rajput, Shiva Valia, Madhav Aggarwal, Aditya Gairola

**Abstract**: Quantum computing is reshaping the security landscape of modern telecommunications. The cryptographic foundations that secure todays 5G systems, including RSA, Elliptic Curve Cryptography (ECC), and Diffie-Hellman (DH), are all susceptible to attacks enabled by Shors algorithm. Protecting 5G networks against future quantum adversaries has therefore become an urgent engineering and research priority. In this paper we introduce QORE, a quantum-secure 5G and Beyond 5G (B5G) Core framework that provides a clear pathway for transitioning both the 5G Core Network Functions and User Equipment (UE) to Post-Quantum Cryptography (PQC). The framework uses the NIST-standardized lattice-based algorithms Module-Lattice Key Encapsulation Mechanism (ML-KEM) and Module-Lattice Digital Signature Algorithm (ML-DSA) and applies them across the 5G Service-Based Architecture (SBA). A Hybrid PQC (HPQC) configuration is also proposed, combining classical and quantum-safe primitives to maintain interoperability during migration. Experimental validation shows that ML-KEM achieves quantum security with minor performance overhead, meeting the low-latency and high-throughput requirements of carrier-grade 5G systems. The proposed roadmap aligns with ongoing 3GPP SA3 and SA5 study activities on the security and management of post-quantum networks as well as with NIST PQC standardization efforts, providing practical guidance for mitigating quantum-era risks while safeguarding long-term confidentiality and integrity of network data.

摘要: 量子计算正在重塑现代电信的安全格局。保护当今5G系统安全的加密基础，包括RSA、椭圆曲线密码学（EC）和迪夫-赫尔曼（DH），都容易受到Shors算法发起的攻击。因此，保护5G网络免受未来量子对手的侵害已成为紧迫的工程和研究优先事项。在本文中，我们介绍了QORE，这是一个量子安全的5G和超越5G（B5 G）核心框架，它为将5G核心网络功能和用户设备（UE）过渡到后量子加密（PQC）提供了明确的途径。该框架使用NIST标准化的基于格的算法模块格密钥封装机制（ML-KEM）和模块格数字签名算法（ML-DSA），并将其应用于5G基于服务的架构（SBA）。还提出了一种混合PQC（HPQC）配置，结合经典和量子安全原语，以保持迁移过程中的互操作性。实验验证表明，ML-KEM以较小的性能开销实现了量子安全，满足了运营商级5G系统的低延迟和高吞吐量要求。拟议的路线图与正在进行的3GPP SA 3和SA 5关于后量子网络安全和管理的研究活动以及NIST PQC标准化工作保持一致，为减轻量子时代的风险提供实用指导，同时保护网络数据的长期机密性和完整性。



## **17. Towards Strong Certified Defense with Universal Asymmetric Randomization**

通过普遍不对称随机化实现强大的认证防御 cs.LG

Accepted by CSF 2026, 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19977v1) [paper-pdf](http://arxiv.org/pdf/2510.19977v1)

**Authors**: Hanbin Hong, Ashish Kundu, Ali Payani, Binghui Wang, Yuan Hong

**Abstract**: Randomized smoothing has become essential for achieving certified adversarial robustness in machine learning models. However, current methods primarily use isotropic noise distributions that are uniform across all data dimensions, such as image pixels, limiting the effectiveness of robustness certification by ignoring the heterogeneity of inputs and data dimensions. To address this limitation, we propose UCAN: a novel technique that \underline{U}niversally \underline{C}ertifies adversarial robustness with \underline{A}nisotropic \underline{N}oise. UCAN is designed to enhance any existing randomized smoothing method, transforming it from symmetric (isotropic) to asymmetric (anisotropic) noise distributions, thereby offering a more tailored defense against adversarial attacks. Our theoretical framework is versatile, supporting a wide array of noise distributions for certified robustness in different $\ell_p$-norms and applicable to any arbitrary classifier by guaranteeing the classifier's prediction over perturbed inputs with provable robustness bounds through tailored noise injection. Additionally, we develop a novel framework equipped with three exemplary noise parameter generators (NPGs) to optimally fine-tune the anisotropic noise parameters for different data dimensions, allowing for pursuing different levels of robustness enhancements in practice.Empirical evaluations underscore the significant leap in UCAN's performance over existing state-of-the-art methods, demonstrating up to $182.6\%$ improvement in certified accuracy at large certified radii on MNIST, CIFAR10, and ImageNet datasets.\footnote{Code is anonymously available at \href{https://github.com/youbin2014/UCAN/}{https://github.com/youbin2014/UCAN/}}

摘要: 随机平滑对于在机器学习模型中实现经过认证的对抗鲁棒性至关重要。然而，当前的方法主要使用在所有数据维度（例如图像像素）上均匀的各向同性噪音分布，从而通过忽略输入和数据维度的同质性来限制稳健性认证的有效性。为了解决这一限制，我们提出了UCAN：一种新型技术，通过\underworldly\underworldly\underworldly {C}通过\underworlded {A} nentropic\underworlded {N}oise来证明对抗鲁棒性。UCAN旨在增强任何现有的随机平滑方法，将其从对称（各向同性）噪音分布转换为不对称（各向异性）噪音分布，从而提供更定制的防御对抗攻击。我们的理论框架是通用的，支持广泛的噪音分布，以在不同的$\ell_p$-规范中获得认证的鲁棒性，并且通过定制的噪音注入保证分类器对具有可证明的鲁棒性边界的扰动输入的预测，适用于任何任意分类器。此外，我们开发了一个配备三个示例性噪音参数发生器（NPG）的新型框架，以最佳方式微调不同数据维度的各向异性噪音参数，从而在实践中追求不同水平的鲁棒性增强。经验评估强调了UCAN性能相对于现有最先进方法的显着飞跃，在MNIST、CIFAR 10和ImageNet数据集上，在大认证半径下，认证准确性提高了高达182.6美元。\脚注{代码可在\href{https：//github.com/youbin2014/UCAN/}{https：//github.com/youbin2014/UCAN/}}



## **18. Q-RAN: Quantum-Resilient O-RAN Architecture**

Q-RAN：量子弹性O-RAN架构 cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19968v1) [paper-pdf](http://arxiv.org/pdf/2510.19968v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Madhav Agarwal, Nitin Rajput, Kriish Sharma, Sushant Mundepi, Shivam Gangwar, Rudraksh Rawal, Jishan

**Abstract**: The telecommunications industry faces a dual transformation: the architectural shift toward Open Radio Access Networks (O-RAN) and the emerging threat from quantum computing. O-RAN disaggregated, multi-vendor architecture creates a larger attack surface vulnerable to crypt-analytically relevant quantum computers(CRQCs) that will break current public key cryptography. The Harvest Now, Decrypt Later (HNDL) attack strategy makes this threat immediate, as adversaries can intercept encrypted data today for future decryption. This paper presents Q-RAN, a comprehensive quantum-resistant security framework for O-RAN networks using NIST-standardized Post-Quantum Cryptography (PQC). We detail the implementation of ML-KEM (FIPS 203) and ML-DSA (FIPS 204), integrated with Quantum Random Number Generators (QRNG) for cryptographic entropy. The solution deploys PQ-IPsec, PQ-DTLS, and PQ-mTLS protocols across all O-RAN interfaces, anchored by a centralized Post-Quantum Certificate Authority (PQ-CA) within the SMO framework. This work provides a complete roadmap for securing disaggregated O-RAN ecosystems against quantum adversaries.

摘要: 电信行业面临双重转型：向开放式无线电接入网络（O-RAN）的架构转变以及来自量子计算的新威胁。O-RAN分解的多供应商架构创建了更大的攻击面，容易受到密码分析相关量子计算机（CRQC）的攻击，这将打破当前的公钥加密技术。立即收获，稍后解密（HNDL）攻击策略使这种威胁迫在眉睫，因为对手可以立即拦截加密数据，以便将来解密。本文介绍了Q-RAN，这是一个全面的抗量子安全框架，用于O-RAN网络，使用NIH标准化的后量子密码学（PQC）。我们详细介绍了ML-KEM（TIP 203）和ML-DSA（TIP 204）的实现，并与量子随机数发生器（QRNG）集成以实现加密信息的信息。该解决方案跨所有O-RAN接口部署PQ-SYS、PQ-DTLS和PQ-mSSL协议，并由NSO框架内的集中式后量子证书颁发机构（PQ-CA）锚定。这项工作为保护分散的O-RAN生态系统免受量子对手的侵害提供了完整的路线图。



## **19. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

未学习但未被遗忘：LLM中精确未学习后的数据提取 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。



## **20. Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?**

现代语音增强系统容易受到对抗攻击吗？ eess.AS

Copyright 2026 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21087v2) [paper-pdf](http://arxiv.org/pdf/2509.21087v2)

**Authors**: Rostislav Makarov, Lea Schönherr, Timo Gerkmann

**Abstract**: Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.

摘要: 用于语音增强的机器学习方法正变得越来越具有表达力，从而能够对输入信号进行更强大的修改。在本文中，我们证明了这种表现力引入了一个漏洞：高级语音增强模型可能容易受到对抗性攻击。具体来说，我们表明，可以注入经过精心设计并由原始输入在心理声学上掩盖的对抗性噪音，以便增强的语音输出传达完全不同的语义含义。我们通过实验验证了当代预测语音增强模型确实可以以这种方式操纵。此外，我们强调，具有随机采样器的扩散模型通过设计表现出对此类对抗性攻击的固有鲁棒性。



## **21. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

关于在异类网络中扩展LT编码区块链及其对拒绝服务威胁的脆弱性 cs.IT

To appear in Future Generation Computer Systems, 2025. This is an  extended version of a shorter version that has appeared in IEEE ICC 2024

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2402.05620v3) [paper-pdf](http://arxiv.org/pdf/2402.05620v3)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.

摘要: 编码区块链作为降低存储成本和促进可扩展性的有希望的解决方案而受到重视。在这一类中，卢比变换（LT）编码区块链是一个有吸引力的可扩展性选择，因为它可以提供广泛的低复杂度解码器。在本工作的第一部分中，我们发现传统的LT解码器（如Belief Propagation和On-the-Fly Gaussian Elimation）对于节点具有不同计算和下载能力的异类网络来说可能不是最佳选择。为了解决这个问题，我们引入了一系列LT代码的混合解码器，并为它们提出了最佳操作机制，以便以最低的解码成本恢复区块链。虽然人们从存储节省和可扩展性方面研究了LT编码的区块链架构，但对其安全漏洞知之甚少。针对这一研究空白，在第二部分中，我们在LT编码区块链上提出了新型的拒绝服务威胁，这些威胁针对具有特定解码能力的节点，阻止它们加入网络。我们提出的威胁本质上是非无意识的，其中对手可以访问存档的块，并根据底层编码方案选择对其中的子集执行攻击。我们表明，我们的优化威胁可以在有限的资源下实现与盲目攻击相同程度的破坏。总的来说，这是第一个此类工作，它提出了设计编码区块链的新问题，以共同提供存储节省、可扩展性以及对优化威胁的弹性。



## **22. Exploring the Effect of DNN Depth on Adversarial Attacks in Network Intrusion Detection Systems**

探讨DNN深度对网络入侵检测系统中对抗攻击的影响 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19761v1) [paper-pdf](http://arxiv.org/pdf/2510.19761v1)

**Authors**: Mohamed ElShehaby, Ashraf Matrawy

**Abstract**: Adversarial attacks pose significant challenges to Machine Learning (ML) systems and especially Deep Neural Networks (DNNs) by subtly manipulating inputs to induce incorrect predictions. This paper investigates whether increasing the layer depth of deep neural networks affects their robustness against adversarial attacks in the Network Intrusion Detection System (NIDS) domain. We compare the adversarial robustness of various deep neural networks across both \ac{NIDS} and computer vision domains (the latter being widely used in adversarial attack experiments). Our experimental results reveal that in the NIDS domain, adding more layers does not necessarily improve their performance, yet it may actually significantly degrade their robustness against adversarial attacks. Conversely, in the computer vision domain, adding more layers exhibits a more modest impact on robustness. These findings can guide the development of robust neural networks for (NIDS) applications and highlight the unique characteristics of network security domains within the (ML) landscape.

摘要: 对抗性攻击通过巧妙地操纵输入来引发错误的预测，对机器学习（ML）系统，尤其是深度神经网络（DNN）构成了重大挑战。本文研究了增加深度神经网络的层深度是否会影响其在网络入侵检测系统（NIDS）领域中对抗性攻击的鲁棒性。我们比较了\ac{NIDS}和计算机视觉领域（后者被广泛用于对抗攻击实验）的各种深度神经网络的对抗鲁棒性。我们的实验结果表明，在NIDS领域，添加更多层并不一定会提高它们的性能，但实际上可能会显着降低它们对对抗攻击的鲁棒性。相反，在计算机视觉领域，添加更多层对稳健性的影响更为温和。这些发现可以指导（NIDS）应用程序的稳健神经网络的开发，并强调（ML）环境中网络安全域的独特特征。



## **23. Explainable Face Presentation Attack Detection via Ensemble-CAM**

通过Ensemble-CAM检测可解释的面部呈现攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19695v1) [paper-pdf](http://arxiv.org/pdf/2510.19695v1)

**Authors**: Rashik Shadman, M G Sarwar Murshed, Faraz Hussain

**Abstract**: Presentation attacks represent a critical security threat where adversaries use fake biometric data, such as face, fingerprint, or iris images, to gain unauthorized access to protected systems. Various presentation attack detection (PAD) systems have been designed leveraging deep learning (DL) models to mitigate this type of threat. Despite their effectiveness, most of the DL models function as black boxes - their decisions are opaque to their users. The purpose of explainability techniques is to provide detailed information about the reason behind the behavior or decision of DL models. In particular, visual explanation is necessary to better understand the decisions or predictions of DL-based PAD systems and determine the key regions due to which a biometric image is considered real or fake by the system. In this work, a novel technique, Ensemble-CAM, is proposed for providing visual explanations for the decisions made by deep learning-based face PAD systems. Our goal is to improve DL-based face PAD systems by providing a better understanding of their behavior. Our provided visual explanations will enhance the transparency and trustworthiness of DL-based face PAD systems.

摘要: 演示攻击是一种严重的安全威胁，对手使用虚假生物识别数据（例如面部、指纹或虹膜图像）来未经授权地访问受保护的系统。已经设计了各种演示攻击检测（PAD）系统，利用深度学习（DL）模型来缓解此类威胁。尽管它们有效，但大多数DL模型都像黑匣子一样发挥作用--它们的决定对用户来说是不透明的。可解释性技术的目的是提供有关DL模型行为或决策背后原因的详细信息。特别是，视觉解释对于更好地理解基于DL的PAD系统的决策或预测并确定系统认为生物识别图像真实或虚假的关键区域是必要的。在这项工作中，提出了一种新颖的技术Ensemble-CAM，用于为基于深度学习的面部DPP系统做出的决策提供视觉解释。我们的目标是通过更好地了解基于DL的面部DPP系统的行为来改进其。我们提供的视觉解释将增强基于DL的面部DPP系统的透明度和可信度。



## **24. Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent**

风格攻击伪装：当字体成为对抗意图的伪装 cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19641v1) [paper-pdf](http://arxiv.org/pdf/2510.19641v1)

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation.

摘要: 随着社交媒体的发展，用户使用风格字体和类似字体的表情符号来表达个性，创建视觉吸引力且保持人类可读的文本。然而，这些字体在NLP模型中引入了隐藏的漏洞：虽然人类很容易阅读风格文本，但模型将这些字符作为不同的标记处理，从而造成干扰。我们识别了这种人类模型的感知差距，并提出了一种基于风格的攻击，即风格攻击伪装（SAD）。我们设计了两种尺寸：轻型用于查询效率，重型用于卓越的攻击性能。跨传统模型、LLM和商业服务的情感分类和机器翻译实验证明了SAD的强大攻击性能。我们还展示了SAD对多模式任务（包括文本到图像和文本到语音生成）的潜在威胁。



## **25. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

你能相信你所看到的吗？Alpha通道对视频对象检测的无框攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.

摘要: 随着对象检测模型越来越多地部署在自动驾驶汽车（AV）和监控平台等网络物理系统中，确保其针对对抗威胁的安全性至关重要。虽然之前的工作探讨了图像领域中的对抗攻击，但视频领域中的这些攻击在很大程度上仍然没有得到审查，尤其是在无框环境中。在本文中，我们介绍了{\Alpha}-Cloak，这是对对象检测器的第一个无箱对抗攻击，完全通过RGBA视频的Alpha通道进行操作。{\Alpha}-Cloak利用Alpha通道将恶意目标视频与良性视频融合，导致融合后的视频对人类观看者来说似乎无害，但始终欺骗对象检测器。我们的攻击不需要访问模型架构、参数或输出，并且不会引入可感知的伪影。我们系统性地研究了对常见视频格式和播放应用程序中Alpha通道的支持，并设计了一种融合算法来确保视觉隐形性和兼容性。我们在五个最先进的对象检测器、视觉语言模型和多模式大型语言模型（Gemini-2.0-Flash）上评估了{\Alpha}-Cloak，展示了在所有场景下100%的攻击成功率。我们的研究结果揭示了基于视频的感知系统中以前未探索的漏洞，凸显了对对抗环境中阿尔法通道的防御的迫切需要。



## **26. A New Type of Adversarial Examples**

新型对抗性例子 cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19347v1) [paper-pdf](http://arxiv.org/pdf/2510.19347v1)

**Authors**: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang

**Abstract**: Most machine learning models are vulnerable to adversarial examples, which poses security concerns on these models. Adversarial examples are crafted by applying subtle but intentionally worst-case modifications to examples from the dataset, leading the model to output a different answer from the original example. In this paper, adversarial examples are formed in an exactly opposite manner, which are significantly different from the original examples but result in the same answer. We propose a novel set of algorithms to produce such adversarial examples, including the negative iterative fast gradient sign method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM), along with their momentum variants: the negative momentum iterative fast gradient sign method (NMI-FGSM) and the negative momentum iterative fast gradient method (NMI-FGM). Adversarial examples constructed by these methods could be used to perform an attack on machine learning systems in certain occasions. Moreover, our results show that the adversarial examples are not merely distributed in the neighbourhood of the examples from the dataset; instead, they are distributed extensively in the sample space.

摘要: 大多数机器学习模型都容易受到对抗性示例的影响，这对这些模型带来了安全问题。对抗性示例是通过对数据集中的示例应用微妙但故意最坏情况的修改来精心设计的，导致模型输出与原始示例不同的答案。在本文中，对抗性例子以完全相反的方式形成，与原始例子显着不同，但得到相同的答案。我们提出了一组新颖的算法来产生此类对抗性示例，包括负迭代快速梯度符号法（NI-FGSM）和负迭代快速梯度法（NI-FGM），以及它们的动量变体：负动量迭代快速梯度符号法（NMI-FGSM）和负动量迭代快速梯度法（NMI-FGM）。通过这些方法构建的对抗性示例可以在某些情况下用于对机器学习系统进行攻击。此外，我们的结果表明，对抗性示例不仅分布在数据集示例的附近;相反，它们广泛分布在样本空间中。



## **27. Collaborative penetration testing suite for emerging generative AI algorithms**

针对新兴生成式人工智能算法的协作渗透测试套件 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19303v1) [paper-pdf](http://arxiv.org/pdf/2510.19303v1)

**Authors**: Petar Radanliev

**Abstract**: Problem Space: AI Vulnerabilities and Quantum Threats Generative AI vulnerabilities: model inversion, data poisoning, adversarial inputs. Quantum threats Shor Algorithm breaking RSA ECC encryption. Challenge Secure generative AI models against classical and quantum cyberattacks. Proposed Solution Collaborative Penetration Testing Suite Five Integrated Components: DAST SAST OWASP ZAP, Burp Suite, SonarQube, Fortify. IAST Contrast Assess integrated with CI CD pipeline. Blockchain Logging Hyperledger Fabric for tamper-proof logs. Quantum Cryptography Lattice based RLWE protocols. AI Red Team Simulations Adversarial ML & Quantum-assisted attacks. Integration Layer: Unified workflow for AI, cybersecurity, and quantum experts. Key Results 300+ vulnerabilities identified across test environments. 70% reduction in high-severity issues within 2 weeks. 90% resolution efficiency for blockchain-logged vulnerabilities. Quantum-resistant cryptography maintained 100% integrity in tests. Outcome: Quantum AI Security Protocol integrating Blockchain Quantum Cryptography AI Red Teaming.

摘要: 问题空间：人工智能漏洞和量子威胁生成人工智能漏洞：模型倒置、数据中毒、对抗性输入。量子威胁Shor算法破解RSA椭圆曲线加密。挑战确保生成性人工智能模型免受经典和量子网络攻击。提议的解决方案协作渗透测试套件五个集成组件：DAST SAST OWASP ZAP、Burp Suite、SonarQube、Fortify。IAST对比评估与CI CD管道集成。区块链日志Hyperledger结构，用于防篡改日志。基于量子密码格子的RLWE协议。AI Red Team模拟对抗ML和量子辅助攻击。集成层：针对人工智能、网络安全和量子专家的统一工作流程。关键结果测试环境中发现了300多个漏洞。2周内高严重性问题减少70%。区块链记录漏洞的解决效率为90%。抗量子加密技术在测试中保持了100%的完整性。结果：集成区块链量子加密技术的量子AI安全协议AI Red团队。



## **28. Adversarial Attacks on LiDAR-Based Tracking Across Road Users: Robustness Evaluation and Target-Aware Black-Box Method**

对基于LiDART的跨路跟踪用户的对抗攻击：稳健性评估和目标感知黑匣子方法 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2410.20893v3) [paper-pdf](http://arxiv.org/pdf/2410.20893v3)

**Authors**: Shengjing Tian, Xiantong Zhao, Yuhao Bian, Yinan Han, Bin Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDART点云跟踪模型在对抗性攻击下的稳健性，这是一个经常被忽视的关键方面，而有利于性能增强。尽管这些模型采用了Transformer或Bird ' s Eye View（BEV）等先进架构，但在面对对抗性攻击、域转移或数据损坏等挑战时，往往会忽视稳健性。相反，我们关注的是跟踪模型在对抗攻击威胁下的稳健性。我们首先建立一个统一的框架，用于在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑匣子攻击策略。对于白盒攻击，我们定制特定的损失函数以适应各种跟踪范式，并将FGSM、C\& W和PVD等现有方法扩展到点云域。在解决黑匣子攻击场景时，我们引入了一种新颖的基于传输的方法，即目标感知扰动生成（TAPG）算法，其双重目标是实现高攻击性能和保持低感知性。该方法采用启发式策略来强制稀疏攻击约束，并利用随机子载体分解来增强可移植性。我们的实验结果揭示了高级跟踪方法在遭受黑匣子和白盒攻击时存在重大漏洞，强调了将对抗攻击的鲁棒性纳入LiDART点云跟踪模型的设计中的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和干扰的隐藏之间取得了最佳平衡。



## **29. Defending Against Prompt Injection with DataFilter**

使用数据过滤器防御提示注入 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.

摘要: 当大型语言模型（LLM）代理越来越多地被部署来自动化任务并与不受信任的外部数据交互时，即时注入成为一个重大的安全威胁。通过将恶意指令注入LLM访问的数据中，攻击者可以任意覆盖原始用户任务，并将代理重定向到无意的、可能有害的操作。现有的防御要么需要访问模型权重（微调），导致大量效用损失（基于检测），要么要求进行非平凡的系统重新设计（系统级）。出于此动机，我们提出了数据过滤器，这是一种测试时模型不可知的防御，可以在数据到达后台LLM之前从数据中删除恶意指令。数据过滤器通过模拟注入的监督微调进行训练，并利用用户的指令和数据来选择性地剥离对抗内容，同时保留良性信息。在多个基准测试中，数据过滤器一致地将即时注入攻击成功率降低到接近零，同时保持LLM的实用性。数据过滤器提供强大的安全性、高实用性和即插即用部署，使其成为保护黑匣子商业LLM免受即时注入的强大实用防御。我们的数据过滤器模型已在https://huggingface.co/JoyYizhu/DataFilter上发布，可立即使用，其代码可在https://github.com/yizhu-joy/DataFilter上重现我们的结果。



## **30. FeatureFool: Zero-Query Fooling of Video Models via Feature Map**

DeliverureFool：通过特征图对视频模型进行零查询愚弄 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.18362v2) [paper-pdf](http://arxiv.org/pdf/2510.18362v2)

**Authors**: Duoxun Tang, Xi Xiao, Guangwu Hu, Kangkang Sun, Xiao Yang, Dongyang Chen, Qing Li, Yongjie Yin, Jiyao Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.

摘要: 深度神经网络（DNN）的脆弱性已初步得到验证。现有的黑匣子对抗攻击通常需要与模型进行多轮交互并消耗大量查询，这在现实世界中是不切实际的，并且很难扩展到最近出现的Video-LLM。此外，视频领域中没有任何攻击直接利用特征地图来移动干净视频特征空间。因此，我们提出了DeliverureFool，这是一种隐形的、视频域的、零查询黑匣子攻击，利用从DNN提取的信息来改变干净视频的特征空间。与依赖于迭代交互的基于查询的方法不同，InspectureFool通过直接利用DNN提取的信息来执行零查询攻击。这种高效的方法在视频领域是前所未有的。实验表明，在没有任何查询的情况下，DeliverureFool针对传统视频分类器的攻击成功率超过70%。受益于特征地图的可移植性，它还可以制作有害内容并绕过Video-LLM识别。此外，DeliverureFool生成的对抗视频在SSIM、PSNR和时间不一致性方面表现出高质量，使攻击几乎不可察觉。本文可能包含暴力或露骨内容。



## **31. The Black Tuesday Attack: how to crash the stock market with adversarial examples to financial forecasting models**

黑色星期二攻击：如何通过金融预测模型的对抗例子来崩溃股市 cs.CR

15 pages, 2 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18990v1) [paper-pdf](http://arxiv.org/pdf/2510.18990v1)

**Authors**: Thomas Hofweber, Jefrey Bergl, Ian Reyes, Amir Sadovnik

**Abstract**: We investigate and defend the possibility of causing a stock market crash via small manipulations of individual stock values that together realize an adversarial example to financial forecasting models, causing these models to make the self-fulfilling prediction of a crash. Such a crash triggered by an adversarial example would likely be hard to detect, since the model's predictions would be accurate and the interventions that would cause it are minor. This possibility is a major risk to financial stability and an opportunity for hostile actors to cause great economic damage to an adversary. This threat also exists against individual stocks and the corresponding valuation of individual companies. We outline how such an attack might proceed, what its theoretical basis is, how it can be directed towards a whole economy or an individual company, and how one might defend against it. We conclude that this threat is vastly underappreciated and requires urgent research on how to defend against it.

摘要: 我们调查并捍卫通过对个别股票价值的小额操纵而导致股市崩盘的可能性，这些操纵共同实现了金融预测模型的对抗性例子，导致这些模型对崩盘做出自我实现的预测。由敌对例子引发的这种崩溃可能很难检测到，因为模型的预测是准确的，而且导致崩溃的干预措施也很小。这种可能性是金融稳定的重大风险，也是敌对行为者对对手造成巨大经济损害的机会。这种威胁也存在于个别股票和个别公司的相应估值。我们概述了这种攻击可能如何进行、其理论基础是什么、如何针对整个经济或单个公司以及如何防御它。我们的结论是，这种威胁被严重低估，需要紧急研究如何防御它。



## **32. Towards Universal Solvers: Using PGD Attack in Active Learning to Increase Generalizability of Neural Operators as Knowledge Distillation from Numerical PDE Solvers**

走向通用求解器：在主动学习中使用PVD攻击来提高神经运算符的可概括性，作为数字偏出方程求解器的知识提炼 cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18989v1) [paper-pdf](http://arxiv.org/pdf/2510.18989v1)

**Authors**: Yifei Sun

**Abstract**: Nonlinear PDE solvers require fine space-time discretizations and local linearizations, leading to high memory cost and slow runtimes. Neural operators such as FNOs and DeepONets offer fast single-shot inference by learning function-to-function mappings and truncating high-frequency components, but they suffer from poor out-of-distribution (OOD) generalization, often failing on inputs outside the training distribution. We propose an adversarial teacher-student distillation framework in which a differentiable numerical solver supervises a compact neural operator while a PGD-style active sampling loop searches for worst-case inputs under smoothness and energy constraints to expand the training set. Using differentiable spectral solvers enables gradient-based adversarial search and stabilizes sample mining. Experiments on Burgers and Navier-Stokes systems demonstrate that adversarial distillation substantially improves OOD robustness while preserving the low parameter cost and fast inference of neural operators.

摘要: 非线性偏东方程求解器需要精细的时空离散化和局部线性化，从而导致内存成本高和运行时间慢。FNOs和DeepONets等神经运算符通过学习函数到函数映射和截断高频分量来提供快速的单次推理，但它们的分布外（OOD）概括性较差，通常在训练分布之外的输入上失败。我们提出了一个对抗性师生蒸馏框架，其中可微数值求解器监督紧凑的神经操作员，而PGD式主动采样循环在平滑度和能量约束下搜索最坏情况的输入以扩展训练集。使用可微谱求解器可以实现基于梯度的对抗搜索并稳定样本挖掘。Burgers和Navier-Stokes系统上的实验表明，对抗蒸馏大大提高了OOD鲁棒性，同时保持了低参数成本和神经操作符的快速推理。



## **33. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

NEXUS：在多轮LLM越狱中利用不安全序列的网络探索 cs.CR

This paper has been accepted in the main conference proceedings of  the 2025 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.03417v2) [paper-pdf](http://arxiv.org/pdf/2510.03417v2)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但仍然容易受到越狱攻击，特别是在良性交换中散布恶意意图并绕过对齐机制的多回合越狱。现有的方法常常无法很好地探索对抗空间，依赖于手工制作的启发式方法，或者缺乏系统性的查询细化。我们介绍了NEXUS（用于eXploiting Unsafe Sequences的网络探索），这是一个用于构建、细化和执行优化多回合攻击的模块化框架。NEXUS包括：（1）IncreghtNet，它将有害意图分层扩展到主题、实体和查询链的结构化语义网络中;（2）反馈驱动的模拟器，通过攻击者-受害者-法官LLM协作使用危害性和语义相似性基准来迭代细化和修剪这些链;（3）网络穿越器，自适应地导航细化查询空间以进行实时攻击。该管道揭示了LLC之间隐秘、高成功的对抗路径。在几种闭源和开源LLM上，NEXUS将攻击成功率比之前的方法提高了2.1%至19.4%。代码：https://github.com/inspire-lab/NEXUS



## **34. Nondeterminism-Aware Optimistic Verification for Floating-Point Neural Networks**

浮点神经网络的不确定性意识乐观验证 cs.CR

17 pages, 7 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16028v2) [paper-pdf](http://arxiv.org/pdf/2510.16028v2)

**Authors**: Jianzhu Yao, Hongxu Su, Taobo Liao, Zerui Cheng, Huan Zhang, Xuechao Wang, Pramod Viswanath

**Abstract**: Neural networks increasingly run on hardware outside the user's control (cloud GPUs, inference marketplaces). Yet ML-as-a-Service reveals little about what actually ran or whether returned outputs faithfully reflect the intended inputs. Users lack recourse against service downgrades (model swaps, quantization, graph rewrites, or discrepancies like altered ad embeddings). Verifying outputs is hard because floating-point(FP) execution on heterogeneous accelerators is inherently nondeterministic. Existing approaches are either impractical for real FP neural networks or reintroduce vendor trust. We present NAO: a Nondeterministic tolerance Aware Optimistic verification protocol that accepts outputs within principled operator-level acceptance regions rather than requiring bitwise equality. NAO combines two error models: (i) sound per-operator IEEE-754 worst-case bounds and (ii) tight empirical percentile profiles calibrated across hardware. Discrepancies trigger a Merkle-anchored, threshold-guided dispute game that recursively partitions the computation graph until one operator remains, where adjudication reduces to a lightweight theoretical-bound check or a small honest-majority vote against empirical thresholds. Unchallenged results finalize after a challenge window, without requiring trusted hardware or deterministic kernels. We implement NAO as a PyTorch-compatible runtime and a contract layer currently deployed on Ethereum Holesky testnet. The runtime instruments graphs, computes per-operator bounds, and runs unmodified vendor kernels in FP32 with negligible overhead (0.3% on Qwen3-8B). Across CNNs, Transformers and diffusion models on A100, H100, RTX6000, RTX4090, empirical thresholds are $10^2-10^3$ times tighter than theoretical bounds, and bound-aware adversarial attacks achieve 0% success. NAO reconciles scalability with verifiability for real-world heterogeneous ML compute.

摘要: 神经网络越来越多地运行在用户控制之外的硬件上（云图形处理器、推理市场）。然而，ML即服务几乎没有透露实际运行的内容或返回的输出是否忠实地反映预期输入。用户缺乏对服务降级（模型交换、量化、图形重写或改变广告嵌入等差异）的追索权。收件箱输出很困难，因为异类加速器上的浮点（FP）执行本质上是不确定的。现有的方法要么对于真正的FP神经网络来说不切实际，要么重新引入供应商信任。我们介绍NAO：一种不确定性容忍感知乐观验证协议，接受原则操作员级接受区域内的输出，而不是要求按位相等。NAO结合了两个错误模型：（i）良好的每个操作员IEEE-754最坏情况界限和（ii）跨硬件校准的严格经验百分位数轮廓。离散性触发默克锚定、阈值引导的争议游戏，该游戏将计算图循环分割，直到剩下一个操作员，其中裁决简化为轻量级的理论约束检查或针对经验阈值的少量诚实多数投票。未挑战的结果在挑战窗口后最终确定，无需受信任的硬件或确定性内核。我们将NAO实现为与PyTorch兼容的运行时和目前部署在以太坊Holesky测试网上的合同层。运行时测量图形、计算每个操作员的界限，并在FP 32中运行未修改的供应商内核，而开销可忽略不计（Qwen 3 -8B上为0.3%）。在A100、H100、RTX 6000、RTX 4090上的CNN、Transformers和扩散模型中，经验阈值比理论界限严格102 - 103 $倍，并且边界感知的对抗性攻击取得了0%的成功。NAO将可扩展性与现实世界的异类ML计算的可验证性相协调。



## **35. HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models**

HarmNet：对大型语言模型进行自适应多回合越狱攻击的框架 cs.CR

This paper has been accepted for presentation at the Conference on  Applied Machine Learning in Information Security (CAMLIS 2025)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18728v1) [paper-pdf](http://arxiv.org/pdf/2510.18728v1)

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement.

摘要: 大型语言模型（LLM）仍然容易受到多回合越狱攻击。我们引入了HarmNet，这是一个模块化框架，包括分层语义网络InghtNet;用于迭代查询细化的反馈驱动模拟器;以及用于实时自适应攻击执行的Network Traverser。HarmNet系统性地探索和完善对抗空间，以发现隐秘、高成功的攻击路径。跨闭源和开源LLM的实验表明，HarmNet优于最先进的方法，实现了更高的攻击成功率。例如，在Mistral-7 B上，HarmNet的攻击成功率达到了99.4%，比最佳基线高出13.9%。索引术语：越狱攻击;大型语言模型;对抗性框架;查询细化。



## **36. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

审判与信任：以全面的防御战略应对拜占庭袭击 cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2505.07614v4) [paper-pdf](http://arxiv.org/pdf/2505.07614v4)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.

摘要: 机器学习的最新进展提高了性能，同时也增加了计算需求。虽然联邦和分布式设置可以解决这些问题，但其结构很容易受到恶意影响。在本文中，我们解决了一个特定的威胁，即拜占庭攻击，其中受影响的客户端注入对抗性更新以破坏全球融合。我们将信任分数概念与尝试函数方法相结合，以动态过滤离群值。我们的方法解决了以前方法的关键局限性，即使在拜占庭节点占多数时也允许功能。此外，我们的算法适用于Adam和RMSProp等广泛使用的缩放方法，以及实际场景，包括本地训练和部分参与。我们通过对从医疗机构收集的合成和真实心电图数据进行广泛的实验来验证我们方法的稳健性。此外，我们还对算法及其对上述实际设置的扩展进行了广泛的理论分析。我们方法的收敛保证与没有拜占庭干扰而开发的经典算法的收敛保证相当。



## **37. Exploring Membership Inference Vulnerabilities in Clinical Large Language Models**

探索临床大型语言模型中的隶属推理漏洞 cs.CR

Accepted at the 1st IEEE Workshop on Healthcare and Medical Device  Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18674v1) [paper-pdf](http://arxiv.org/pdf/2510.18674v1)

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems.

摘要: 随着大型语言模型（LLM）越来越嵌入临床决策支持、文档和患者信息系统中，确保其隐私和可信度已成为医疗保健行业的一个紧迫挑战。对敏感电子健康记录（EHR）数据进行微调LLM可以改善域对齐，但也会增加通过模型行为暴露患者信息的风险。在这项正在进行的工作中，我们对临床LLM中的隶属关系推断漏洞进行了一项探索性实证研究，重点关注对手是否可以推断模型训练期间是否使用了特定的患者记录。使用最先进的临床问答模型Llemr，我们评估了典型的基于损失的攻击和更真实地反映临床对抗状况的基于领域动机的基于重述的扰动策略。我们的初步研究结果揭示了有限但可测量的会员泄露，这表明当前的临床LLM提供了部分抵抗力，但仍然容易受到微妙的隐私风险，这可能会破坏人们对临床人工智能采用的信任。这些结果激励了上下文感知、特定领域隐私评估和防御的持续发展，例如差异隐私微调和转述感知训练，以加强医疗保健人工智能系统的安全性和可信度。



## **38. Quantifying Security for Networked Control Systems: A Review**

网络控制系统安全性量化研究综述 eess.SY

Journal submission

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18645v1) [paper-pdf](http://arxiv.org/pdf/2510.18645v1)

**Authors**: Sribalaji C. Anand, Anh Tung Nguyen, André M. H. Teixeira, Henrik Sandberg, Karl H. Johansson

**Abstract**: Networked Control Systems (NCSs) are integral in critical infrastructures such as power grids, transportation networks, and production systems. Ensuring the resilient operation of these large-scale NCSs against cyber-attacks is crucial for societal well-being. Over the past two decades, extensive research has been focused on developing metrics to quantify the vulnerabilities of NCSs against attacks. Once the vulnerabilities are quantified, mitigation strategies can be employed to enhance system resilience. This article provides a comprehensive overview of methods developed for assessing NCS vulnerabilities and the corresponding mitigation strategies. Furthermore, we emphasize the importance of probabilistic risk metrics to model vulnerabilities under adversaries with imperfect process knowledge. The article concludes by outlining promising directions for future research.

摘要: 网络控制系统（NSO）是电网、交通网络和生产系统等关键基础设施的组成部分。确保这些大型NCs针对网络攻击的弹性运行对于社会福祉至关重要。在过去的二十年里，广泛的研究一直集中在开发指标来量化NSO针对攻击的脆弱性。一旦漏洞被量化，就可以采用缓解策略来增强系统的弹性。本文全面概述了为评估NSO漏洞而开发的方法以及相应的缓解策略。此外，我们强调了概率风险指标对流程知识不完善的对手下的漏洞建模的重要性。文章最后概述了未来研究的有希望的方向。



## **39. Qatsi: Stateless Secret Generation via Hierarchical Memory-Hard Key Derivation**

Qatsi：通过分层存储硬密钥推导产生无状态秘密 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18614v1) [paper-pdf](http://arxiv.org/pdf/2510.18614v1)

**Authors**: René Coignard, Anton Rygin

**Abstract**: We present Qatsi, a hierarchical key derivation scheme using Argon2id that generates reproducible cryptographic secrets without persistent storage. The system eliminates vault-based attack surfaces by deriving all secrets deterministically from a single high-entropy master secret and contextual layers. Outputs achieve 103-312 bits of entropy through memory-hard derivation (64-128 MiB, 16-32 iterations) and provably uniform rejection sampling over 7776-word mnemonics or 90-character passwords. We formalize the hierarchical construction, prove output uniformity, and quantify GPU attack costs: $2.4 \times 10^{16}$ years for 80-bit master secrets on single-GPU adversaries under Paranoid parameters (128 MiB memory). The implementation in Rust provides automatic memory zeroization, compile-time wordlist integrity verification, and comprehensive test coverage. Reference benchmarks on Apple M1 Pro (2021) demonstrate practical usability with 544 ms Standard mode and 2273 ms Paranoid mode single-layer derivations. Qatsi targets air-gapped systems and master credential generation where stateless reproducibility outweighs rotation flexibility.

摘要: 我们提出了Qatsi，这是一种使用Argon 2 id的分层密钥推导方案，可以在无需持久存储的情况下生成可复制的加密秘密。该系统通过从单个高熵主秘密和上下文层确定性地获取所有秘密来消除基于保险库的攻击面。输出通过记忆困难推导（64-128 MiB，16-32次迭代）和对7776个单词的助记符或90个字符的密码进行可证明的均匀拒绝采样来实现103-312位的信息。我们形式化了分层结构，证明输出一致性，并量化了图形处理器攻击成本：在Paranoid参数下，单图形处理对手上的80位主秘密需要2.4美元\x 10 ' 16}$年（128 MiB内存）。Rust中的实现提供了自动内存零化、编译时单词列表完整性验证和全面的测试覆盖。Apple M1 Pro（2021）的参考基准演示了544 ms标准模式和2273 ms偏执模式单层派生的实际可用性。Qatsi的目标是空间隙系统和主证书生成，其中无状态可重复性超过了轮换灵活性。



## **40. Evaluating Large Language Models in detecting Secrets in Android Apps**

评估大型语言模型以检测Android应用程序中的秘密 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18601v1) [paper-pdf](http://arxiv.org/pdf/2510.18601v1)

**Authors**: Marco Alecci, Jordan Samhi, Tegawendé F. Bissyandé, Jacques Klein

**Abstract**: Mobile apps often embed authentication secrets, such as API keys, tokens, and client IDs, to integrate with cloud services. However, developers often hardcode these credentials into Android apps, exposing them to extraction through reverse engineering. Once compromised, adversaries can exploit secrets to access sensitive data, manipulate resources, or abuse APIs, resulting in significant security and financial risks. Existing detection approaches, such as regex-based analysis, static analysis, and machine learning, are effective for identifying known patterns but are fundamentally limited: they require prior knowledge of credential structures, API signatures, or training data.   In this paper, we propose SecretLoc, an LLM-based approach for detecting hardcoded secrets in Android apps. SecretLoc goes beyond pattern matching; it leverages contextual and structural cues to identify secrets without relying on predefined patterns or labeled training sets. Using a benchmark dataset from the literature, we demonstrate that SecretLoc detects secrets missed by regex-, static-, and ML-based methods, including previously unseen types of secrets. In total, we discovered 4828 secrets that were undetected by existing approaches, discovering more than 10 "new" types of secrets, such as OpenAI API keys, GitHub Access Tokens, RSA private keys, and JWT tokens, and more.   We further extend our analysis to newly crawled apps from Google Play, where we uncovered and responsibly disclosed additional hardcoded secrets. Across a set of 5000 apps, we detected secrets in 2124 apps (42.5%), several of which were confirmed and remediated by developers after we contacted them. Our results reveal a dual-use risk: if analysts can uncover these secrets with LLMs, so can attackers. This underscores the urgent need for proactive secret management and stronger mitigation practices across the mobile ecosystem.

摘要: 移动应用程序通常嵌入身份验证秘密，例如API密钥、令牌和客户端ID，以与云服务集成。然而，开发人员经常将这些凭据硬编码到Android应用程序中，通过反向工程将其提取。一旦被泄露，对手就可以利用秘密访问敏感数据、操纵资源或滥用API，从而导致重大的安全和财务风险。现有的检测方法，例如基于regex的分析、静态分析和机器学习，对于识别已知模式有效，但从根本上来说是有限的：它们需要凭证结构、API签名或训练数据的先验知识。   在本文中，我们提出了SecretLoc，这是一种基于LLM的方法，用于检测Android应用程序中的硬编码秘密。SecretLoc超越了模式匹配;它利用上下文和结构线索来识别秘密，而无需依赖预定义的模式或标记的训练集。使用文献中的基准数据集，我们证明SecretLoc可以检测基于regex、静态和ML的方法错过的秘密，包括以前未见过的秘密类型。我们总共发现了4828个现有方法未检测到的秘密，发现了10多种“新”类型的秘密，例如OpenAI API密钥、GitHub Access Tokens、RSA私有密钥和JWT令牌等。   我们进一步将分析扩展到Google Play新抓取的应用程序，在其中我们发现并负责任地披露了额外的硬编码秘密。在一组5000个应用程序中，我们检测到了2124个应用程序（42.5%）中的秘密，其中一些在我们联系开发人员后得到了证实和修复。我们的结果揭示了双重用途风险：如果分析师可以通过LLM发现这些秘密，那么攻击者也可以。这凸显了整个移动生态系统迫切需要积极主动的秘密管理和更强有力的缓解措施。



## **41. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection**

SentinelNet：通过基于信用的动态威胁检测保护多代理协作 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16219v2) [paper-pdf](http://arxiv.org/pdf/2510.16219v2)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

摘要: 恶意代理对大型语言模型（LLM）支持的多代理系统（MAS）的可靠性和决策能力构成重大威胁。由于反应式设计或集中式架构可能会引入单点故障，现有的防御往往会出现缺陷。为了应对这些挑战，我们提出了SentinelNet，这是第一个用于主动检测和缓解多代理协作中恶意行为的去中心化框架。SentinelNet为每个代理配备了一个基于信用的检测器，该检测器通过对增强的对抗辩论轨迹进行对比学习进行训练，从而能够通过底部k消除来自主评估消息可信度和动态邻居排名，以抑制恶意通信。为了克服攻击数据的稀缺性，它生成模拟不同威胁的对抗轨迹，确保稳健的训练。MAS基准测试的实验表明，SentinelNet实现了对恶意代理近乎完美的检测，在两轮辩论中接近100%，并从受损的基线恢复了95%的系统准确性。SentinelNet在跨域和攻击模式之间表现出强大的概括性，建立了一种用于保护协作MAS的新颖范式。



## **42. Robustness Verification of Graph Neural Networks Via Lightweight Satisfiability Testing**

基于轻量级可满足性测试的图神经网络鲁棒性验证 cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18591v1) [paper-pdf](http://arxiv.org/pdf/2510.18591v1)

**Authors**: Chia-Hsuan Lu, Tony Tan, Michael Benedikt

**Abstract**: Graph neural networks (GNNs) are the predominant architecture for learning over graphs. As with any machine learning model, and important issue is the detection of adversarial attacks, where an adversary can change the output with a small perturbation of the input. Techniques for solving the adversarial robustness problem - determining whether such an attack exists - were originally developed for image classification, but there are variants for many other machine learning architectures. In the case of graph learning, the attack model usually considers changes to the graph structure in addition to or instead of the numerical features of the input, and the state of the art techniques in the area proceed via reduction to constraint solving, working on top of powerful solvers, e.g. for mixed integer programming. We show that it is possible to improve on the state of the art in structural robustness by replacing the use of powerful solvers by calls to efficient partial solvers, which run in polynomial time but may be incomplete. We evaluate our tool RobLight on a diverse set of GNN variants and datasets.

摘要: 图神经网络（GNN）是图学习的主要架构。与任何机器学习模型一样，重要的问题是对抗性攻击的检测，对手可以通过对输入的微小扰动来改变输出。用于解决对抗鲁棒性问题（确定是否存在此类攻击）的技术最初是为图像分类开发的，但许多其他机器学习架构也有变体。在图学习的情况下，攻击模型通常会考虑除了输入的数字特征之外或代替输入的数字特征，并且该领域的最新技术通过简化到约束求解来进行，在强大的求解器之上工作，例如，对于混合整元规划。我们表明，通过调用高效的部分求解器来取代强大的求解器，可以提高结构鲁棒性的最新水平，部分求解器在多项时间内运行，但可能不完整。我们在一组不同的GNN变体和数据集上评估我们的工具RobLight。



## **43. Towards Quantum Enhanced Adversarial Robustness with Rydberg Reservoir Learning**

利用Rydberg水库学习实现量子增强对抗鲁棒性 quant-ph

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.13473v2) [paper-pdf](http://arxiv.org/pdf/2510.13473v2)

**Authors**: Shehbaz Tariq, Muhammad Talha, Symeon Chatzinotas, Hyundong Shin

**Abstract**: Quantum reservoir computing (QRC) leverages the high-dimensional, nonlinear dynamics inherent in quantum many-body systems for extracting spatiotemporal patterns in sequential and time-series data with minimal training overhead. Although QRC inherits the expressive capabilities associated with quantum encodings, recent studies indicate that quantum classifiers based on variational circuits remain susceptible to adversarial perturbations. In this perspective, we investigate the first systematic evaluation of adversarial robustness in a QRC based learning model. Our reservoir comprises an array of strongly interacting Rydberg atoms governed by a fixed Hamiltonian, which naturally evolves under complex quantum dynamics, producing high-dimensional embeddings. A lightweight multilayer perceptron serves as the trainable readout layer. We utilize the balanced datasets, namely MNIST, Fashion-MNIST, and Kuzushiji-MNIST, as a benchmark for rigorously evaluating the impact of augmenting the quantum reservoir with a Multilayer perceptron (MLP) in white-box adversarial attacks to assess its robustness. We demonstrate that this approach yields significantly higher accuracy than purely classical models across all perturbation strengths tested. This hybrid approach reveals a new source of quantum advantage and provides practical guidance for the secure deployment of machine learning models on quantum-centric supercomputing with near-term hardware.

摘要: 量子储层计算（QRC）利用量子多体系统固有的多维、非线性动力学，以最小的训练负担提取顺序和时间序列数据中的时空模式。尽管QRC继承了与量子编码相关的表达能力，但最近的研究表明，基于变分电路的量子分类器仍然容易受到对抗性扰动的影响。从这个角度来看，我们研究了基于QRC的学习模型中对对抗稳健性的首次系统评估。我们的水库由一系列强相互作用的里德堡原子组成，这些原子由固定的汉密尔顿量控制，该Hamilton量在复杂的量子动力学下自然进化，产生多维嵌入。轻量级的多层感知器充当可训练的读出层。我们利用平衡数据集（即MNIST、Fashion-MNIST和Kuzushiji-MNIST）作为基准，严格评估在白盒对抗攻击中使用多层感知器（MLP）增强量子库的影响，以评估其稳健性。我们证明，在所有测试的扰动强度下，这种方法比纯经典模型产生了显着更高的准确性。这种混合方法揭示了量子优势的新来源，并为在具有近期硬件的以量子为中心的超级计算上安全部署机器学习模型提供了实践指导。



## **44. S2AP: Score-space Sharpness Minimization for Adversarial Pruning**

S2 AP：对抗性修剪的得分空间清晰度最小化 cs.CV

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18381v1) [paper-pdf](http://arxiv.org/pdf/2510.18381v1)

**Authors**: Giorgio Piras, Qi Zhao, Fabio Brau, Maura Pintor, Christian Wressnegger, Battista Biggio

**Abstract**: Adversarial pruning methods have emerged as a powerful tool for compressing neural networks while preserving robustness against adversarial attacks. These methods typically follow a three-step pipeline: (i) pretrain a robust model, (ii) select a binary mask for weight pruning, and (iii) finetune the pruned model. To select the binary mask, these methods minimize a robust loss by assigning an importance score to each weight, and then keep the weights with the highest scores. However, this score-space optimization can lead to sharp local minima in the robust loss landscape and, in turn, to an unstable mask selection, reducing the robustness of adversarial pruning methods. To overcome this issue, we propose a novel plug-in method for adversarial pruning, termed Score-space Sharpness-aware Adversarial Pruning (S2AP). Through our method, we introduce the concept of score-space sharpness minimization, which operates during the mask search by perturbing importance scores and minimizing the corresponding robust loss. Extensive experiments across various datasets, models, and sparsity levels demonstrate that S2AP effectively minimizes sharpness in score space, stabilizing the mask selection, and ultimately improving the robustness of adversarial pruning methods.

摘要: 对抗性修剪方法已经成为压缩神经网络的强大工具，同时保持对对抗性攻击的鲁棒性。这些方法通常遵循三步流水线：（i）预训练鲁棒模型，（ii）选择用于权重修剪的二进制掩码，以及（iii）微调修剪后的模型。为了选择二进制掩码，这些方法通过为每个权重分配重要性分数来最小化鲁棒性损失，然后保留具有最高分数的权重。然而，这种得分空间优化可能会导致鲁棒损失景观中出现尖锐的局部极小值，进而导致不稳定的掩蔽选择，从而降低对抗性修剪方法的鲁棒性。为了克服这个问题，我们提出了一种新颖的对抗性修剪插件方法，称为得分空间清晰度感知对抗性修剪（S2 AP）。通过我们的方法，我们引入了分数空间清晰度最小化的概念，该概念在屏蔽搜索期间通过扰乱重要性分数并最小化相应的鲁棒损失来操作。跨各种数据集、模型和稀疏度水平的广泛实验表明，S2 AP有效地最大限度地降低了分数空间的清晰度，稳定了掩蔽选择，并最终提高了对抗性修剪方法的鲁棒性。



## **45. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟儿抓住了漏洞：在LLM服务系统中揭开定时侧通道 cs.CR

This work was first submitted for review on Sept. 5, 2024, and the  initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version  has accepted for publication by IEEE Transactions on Information Forensics  and Security (TIFS)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2409.20002v5) [paper-pdf](http://arxiv.org/pdf/2409.20002v5)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **46. Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming**

Genesis：LLM Web Agent Red-Teaming不断发展的攻击策略 cs.AI

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18314v1) [paper-pdf](http://arxiv.org/pdf/2510.18314v1)

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.

摘要: 随着大型语言模型（LLM）代理越来越多地自动化复杂的Web任务，它们提高了生产力，同时引入了新的安全风险。然而，关于Web代理攻击的相关研究仍然有限。现有的红色团队方法主要依赖于手动设计的攻击策略或离线训练的静态模型。此类方法无法捕捉Web代理的底层行为模式，因此很难在不同的环境中进行概括。在Web代理攻击中，成功需要不断发现和进化攻击策略。为此，我们提出了Genesis，这是一个由三个模块组成的新颖的代理框架：攻击者、得分者和策略者。攻击者通过将遗传算法与混合策略表示集成来生成对抗注入。评分者评估目标Web代理的响应以提供反馈。策略师从交互日志中动态发现有效的策略，并将其汇编到不断增长的策略库中，然后重新部署该库以增强攻击者的有效性。跨各种Web任务的广泛实验表明，我们的框架发现了新颖的策略，并且始终优于现有的攻击基线。



## **47. Ensuring Robustness in ML-enabled Software Systems: A User Survey**

确保支持ML的软件系统的稳健性：用户调查 cs.SE

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18292v1) [paper-pdf](http://arxiv.org/pdf/2510.18292v1)

**Authors**: Hala Abdelkader, Mohamed Abdelrazek, Priya Rani, Rajesh Vasa, Jean-Guy Schneider

**Abstract**: Ensuring robustness in ML-enabled software systems requires addressing critical challenges, such as silent failures, out-of-distribution (OOD) data, and adversarial attacks. Traditional software engineering practices, which rely on predefined logic, are insufficient for ML components that depend on data and probabilistic decision-making. To address these challenges, we propose the ML-On-Rails protocol, a unified framework designed to enhance the robustness and trustworthiness of ML-enabled systems in production. This protocol integrates key safeguards such as OOD detection, adversarial attack detection, input validation, and explainability. It also includes a model-to-software communication framework using HTTP status codes to enhance transparency in reporting model outcomes and errors. To align our approach with real-world challenges, we conducted a practitioner survey, which revealed major robustness issues, gaps in current solutions, and highlighted how a standardised protocol such as ML-On-Rails can improve system robustness. Our findings highlight the need for more support and resources for engineers working with ML systems. Finally, we outline future directions for refining the proposed protocol, leveraging insights from the survey and real-world applications to continually enhance its effectiveness.

摘要: 确保支持ML的软件系统的健壮性需要解决关键挑战，例如静默故障、分发外（OOD）数据和对抗性攻击。传统的软件工程实践依赖于预定义的逻辑，对于依赖于数据和概率决策的ML组件来说是不够的。为了应对这些挑战，我们提出了ML-On-Rails协议，这是一个统一的框架，旨在提高生产中支持ML的系统的健壮性和可信度。该协议集成了OOD检测、对抗性攻击检测、输入验证和可解释性等关键保护措施。它还包括一个模型到软件的通信框架，使用HTTP状态代码来增强报告模型结果和错误的透明度。为了使我们的方法与现实世界的挑战保持一致，我们进行了一项从业者调查，揭示了当前解决方案中的主要稳健性问题、差距，并强调了ML-On-Rails等标准化协议如何提高系统稳健性。我们的调查结果凸显了使用ML系统的工程师需要更多支持和资源。最后，我们概述了完善拟议协议的未来方向，利用调查和现实应用程序的见解来不断提高其有效性。



## **48. DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents**

DrunkAgent：LLM-Powered Recommender Agent中的隐形内存损坏 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2503.23804v3) [paper-pdf](http://arxiv.org/pdf/2503.23804v3)

**Authors**: Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.   To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.

摘要: 大型语言模型（LLM）驱动的代理越来越多地用于推荐系统（RS）中以实现个性化行为建模，其中记忆机制在使代理能够自主探索、学习和从现实世界的交互中自我进化方面发挥着关键作用。然而，作为上下文存储库的这种机制本质上暴露了潜在对抗操纵的攻击表面。尽管代理RS发挥着核心作用，但面对此类威胁时的稳健性在很大程度上仍然没有得到充分的探索。之前的作品存在语义不匹配或依赖于静态嵌入或预定义的提示，所有这些都不是为动态系统设计的，尤其是为LLM代理的动态内存状态。商业收件箱的黑匣子性质加剧了这一挑战。   为了解决上述问题，在本文中，我们对LLM支持的推荐代理中基于内存的漏洞进行了首次系统性调查，揭示了它们的安全局限性，并指导加强系统弹性和可信度的努力。具体来说，我们提出了一种新颖的黑匣子攻击框架DrunkAgent。DrunkAgent为目标物品促销精心设计了具有语义意义的对抗性文本触发器，并引入了一系列策略，通过破坏交互期间的记忆更新来最大化触发效应。触发器和策略在代理模型上进行了优化，使DrunkAgent具有可转移性和隐蔽性。跨不同代理RS对现实世界数据集进行的广泛实验，包括协作过滤、检索增强和顺序推荐，证明了DrunkAgent的可概括性、可移植性和隐蔽性。



## **49. Transaction Capacity, Security and Latency in Blockchains**

区块链中的交易容量、安全性和延迟 cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2402.10138v2) [paper-pdf](http://arxiv.org/pdf/2402.10138v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: We analyze how secure a block is after the block becomes $k$-deep, i.e., security-latency, for Nakamoto consensus under an exponential network delay model. We provide the fault tolerance and extensive bounds on safety violation probabilities given mining rate, delay rate and confirmation rules. Next, modeling the blockchain system as a batch service queue with exponential network delay, we connect the security-latency analysis to sustainable transaction rate of the queue system. As our model assumes exponential network delay, batch service queue models give a meaningful trade-off between transaction capacity, security and latency. Our results indicate that, by simply picking $k=7$-block confirmation rule in Bitcoin instead of the convention of $k=6$, mining rate, latency and throughput can be increased sixfold with the same safety guarantees. We further consider adversarial attacks on the queue service to hamper the service process. In an extreme scenario, we consider the selfish-mining attack for this purpose and provide the maximum adversarial block ratio in the longest chain under the exponential delay model. The ratio in turn reflects the maximum rate of decrease in the sustainable transaction rate of the queue.

摘要: 我们分析块变得$k$-深度后的安全性如何，即，安全延迟，指数网络延迟模型下的Nakamoto共识。我们在给定挖掘率、延迟率和确认规则的情况下提供了安全违规概率的安全容忍度和广泛界限。接下来，将区块链系统建模为具有指数级网络延迟的批处理服务队列，我们将安全延迟分析与队列系统的可持续交易率联系起来。由于我们的模型假设指数网络延迟，批服务队列模型在事务容量、安全性和延迟之间提供了有意义的权衡。我们的结果表明，通过简单地选择比特币中的$k=7$-块确认规则，而不是$k=6$的惯例，采矿率、延迟和吞吐量可以在相同的安全保证下增加六倍。我们进一步考虑对队列服务的对抗攻击会阻碍服务过程。在极端情况下，我们为此考虑自私挖掘攻击，并在指数延迟模型下提供最长链中的最大对抗区块比。该比率反过来反映了队列可持续交易率的最大下降率。



## **50. Black-Box Evasion Attacks on Data-Driven Open RAN Apps: Tailored Design and Experimental Evaluation**

对数据驱动开放RAN应用程序的黑匣子规避攻击：定制设计和实验评估 cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18160v1) [paper-pdf](http://arxiv.org/pdf/2510.18160v1)

**Authors**: Pranshav Gajjar, Molham Khoja, Abiodun Ganiyu, Marc Juarez, Mahesh K. Marina, Andrew Lehane, Vijay K. Shah

**Abstract**: The impending adoption of Open Radio Access Network (O-RAN) is fueling innovation in the RAN towards data-driven operation. Unlike traditional RAN where the RAN data and its usage is restricted within proprietary and monolithic RAN equipment, the O-RAN architecture opens up access to RAN data via RAN intelligent controllers (RICs), to third-party machine learning (ML) powered applications - rApps and xApps - to optimize RAN operations. Consequently, a major focus has been placed on leveraging RAN data to unlock greater efficiency gains. However, there is an increasing recognition that RAN data access to apps could become a source of vulnerability and be exploited by malicious actors. Motivated by this, we carry out a comprehensive investigation of data vulnerabilities on both xApps and rApps, respectively hosted in Near- and Non-real-time (RT) RIC components of O-RAN. We qualitatively analyse the O-RAN security mechanisms and limitations for xApps and rApps, and consider a threat model informed by this analysis. We design a viable and effective black-box evasion attack strategy targeting O-RAN RIC Apps while accounting for the stringent timing constraints and attack effectiveness. The strategy employs four key techniques: the model cloning algorithm, input-specific perturbations, universal adversarial perturbations (UAPs), and targeted UAPs. This strategy targets ML models used by both xApps and rApps within the O-RAN system, aiming to degrade network performance. We validate the effectiveness of the designed evasion attack strategy and quantify the scale of performance degradation using a real-world O-RAN testbed and emulation environments. Evaluation is conducted using the Interference Classification xApp and the Power Saving rApp as representatives for near-RT and non-RT RICs. We also show that the attack strategy is effective against prominent defense techniques for adversarial ML.

摘要: 开放式无线电接入网络（O-RAN）的即将采用正在推动RAN向数据驱动运营的创新。与传统的RAN不同，其中RAN数据及其使用受到专有和单片RAN设备的限制，O-RAN架构开放了通过RAN智能控制器（RIC）对RAN数据的访问，以及第三方机器学习（ML）驱动的应用程序（rApp和xApp）来优化RAN操作。因此，主要焦点被放在利用RAN数据来实现更大的效率提升上。然而，人们越来越认识到，对应用程序的RAN数据访问可能成为漏洞来源并被恶意行为者利用。受此启发，我们对分别托管在O-RAN的近实时（RT）RIC组件和非实时（RT）RIC组件中的xApp和rApp的数据漏洞进行了全面调查。我们定性分析了xApp和rApp的O-RAN安全机制和限制，并考虑根据此分析提供的威胁模型。我们针对O-RAN RIC应用程序设计了一种可行且有效的黑匣子规避攻击策略，同时考虑到严格的时间限制和攻击有效性。该策略采用了四项关键技术：模型克隆算法、输入特定扰动、通用对抗扰动（UPC）和有针对性的UAP。该策略针对O-RAN系统内xApp和rApp使用的ML模型，旨在降低网络性能。我们验证了设计的规避攻击策略的有效性，并使用现实世界的O-RAN测试台和仿真环境量化了性能下降的规模。使用干扰分类xApp和省电rApp作为近RT和非RT RIC的代表进行评估。我们还表明，该攻击策略对对抗性ML的突出防御技术有效。



