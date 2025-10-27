# Latest Adversarial Attack Papers
**update at 2025-10-27 09:12:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**

PatchGuard：通过视觉变换器和伪异常进行逆向鲁棒异常检测和定位 cs.CV

Accepted to the Conference on Computer Vision and Pattern Recognition  (CVPR) 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.09237v2) [paper-pdf](http://arxiv.org/pdf/2506.09237v2)

**Authors**: Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban

**Abstract**: Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .

摘要: 异常检测（AD）和异常定位（AL）在医学成像和工业监控等要求高可靠性的领域至关重要。然而，由于训练数据的限制，当前的AD和AL方法通常容易受到对抗攻击，训练数据通常只包括正常的、未标记的样本。本研究引入了PatchGuard，这是一种对抗稳健的AD和AL方法，它在基于Vision Transformer（ViT）的架构中将伪异常与定位屏蔽结合起来，以解决这些漏洞。我们首先研究伪异常的基本属性，然后提供增强AD和AL系统对抗鲁棒性所需的注意机制的理论见解。然后，我们介绍了我们的方法，该方法利用前景感知伪异常来克服之前异常感知方法的缺陷。我们的方法将这些精心设计的伪异常样本整合到基于ViT的框架中，并在旨在提高模型稳健性的新型损失函数指导下进行对抗训练，正如我们的理论分析所支持的那样。在成熟的工业和医疗数据集上的实验结果表明，PatchGuard在对抗环境中的表现显着优于之前的方法，在AD中实现了53.2%美元的性能收益，在AL中实现了68.5%美元的性能收益，同时在非对抗环境中还保持了有竞争力的准确性。代码存储库可在https://github.com/rohban-lab/PatchGuard上获取。



## **2. FrameShield: Adversarially Robust Video Anomaly Detection**

Frame Shield：对抗鲁棒的视频异常检测 cs.LG

28 page, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21532v1) [paper-pdf](http://arxiv.org/pdf/2510.21532v1)

**Authors**: Mojtaba Nafez, Mobina Poulaei, Nikan Vasei, Bardia Soltani Moakhar, Mohammad Sabokrou, MohammadHossein Rohban

**Abstract**: Weakly Supervised Video Anomaly Detection (WSVAD) has achieved notable advancements, yet existing models remain vulnerable to adversarial attacks, limiting their reliability. Due to the inherent constraints of weak supervision, where only video-level labels are provided despite the need for frame-level predictions, traditional adversarial defense mechanisms, such as adversarial training, are not effective since video-level adversarial perturbations are typically weak and inadequate. To address this limitation, pseudo-labels generated directly from the model can enable frame-level adversarial training; however, these pseudo-labels are inherently noisy, significantly degrading performance. We therefore introduce a novel Pseudo-Anomaly Generation method called Spatiotemporal Region Distortion (SRD), which creates synthetic anomalies by applying severe augmentations to localized regions in normal videos while preserving temporal consistency. Integrating these precisely annotated synthetic anomalies with the noisy pseudo-labels substantially reduces label noise, enabling effective adversarial training. Extensive experiments demonstrate that our method significantly enhances the robustness of WSVAD models against adversarial attacks, outperforming state-of-the-art methods by an average of 71.0\% in overall AUROC performance across multiple benchmarks. The implementation and code are publicly available at https://github.com/rohban-lab/FrameShield.

摘要: 弱监督视频异常检测（WSVAD）取得了显着的进步，但现有模型仍然容易受到对抗攻击，从而限制了其可靠性。由于弱监督的固有限制，尽管需要帧级预测，但只提供视频级标签，传统的对抗性防御机制（例如对抗性训练）并不有效，因为视频级对抗性扰动通常很弱且不充分。为了解决这一限制，直接从模型生成的伪标签可以实现帧级对抗训练;然而，这些伪标签本质上是有噪的，会显着降低性能。因此，我们引入了一种新型的伪异常生成方法，称为时空区域失真（SRD），该方法通过对正常视频中的局部区域应用严格的增强来创建合成异常，同时保持时间一致性。将这些经过精确注释的合成异常与有噪的伪标签集成起来，大大降低了标签噪音，从而实现了有效的对抗训练。大量实验表明，我们的方法显着增强了WSVAD模型针对对抗性攻击的鲁棒性，在多个基准测试中，总体AUROC性能平均比最先进的方法高出71.0%。该实现和代码可在https://github.com/rohban-lab/FrameShield上公开获取。



## **3. Fundamental Limitations in Pointwise Defences of LLM Finetuning APIs**

LLM微调API逐点防御的基本局限性 cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.14828v2) [paper-pdf](http://arxiv.org/pdf/2502.14828v2)

**Authors**: Xander Davies, Eric Winsor, Alexandra Souly, Tomek Korbak, Robert Kirk, Christian Schroeder de Witt, Yarin Gal

**Abstract**: LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.

摘要: LLM开发人员已实施技术干预，以防止微调滥用攻击，即对手通过使用公共API微调模型来逃避保障措施的攻击。之前的工作已经针对特定微调API防御建立了几次成功的攻击。在这项工作中，我们表明，寻求检测个体有害训练或推理样本（“逐点”检测）的微调API的防御在防止微调攻击的能力方面从根本上受到限制。我们构建了“逐点不可检测”的攻击，这些攻击在良性模型输出（例如语义或语法变体）中重新利用信息，以秘密传输危险知识。我们的攻击仅由毫无可疑的良性样本组成，这些样本可以在微调之前从模型中收集，这意味着训练和推理样本都是单独良性且低困惑性的。我们测试了针对OpenAI微调API的攻击，发现它们成功地引出了有害的多项选择题的答案，并且它们规避了我们设计的成功检测其他微调攻击的增强型监控系统。我们鼓励社区开发防御措施，以解决我们在逐点微调API防御中发现的基本限制。



## **4. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

部分失去控制权下非线性系统的能量弹性 math.OC

22 pages, 4 figures, 1 table

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.07603v3) [paper-pdf](http://arxiv.org/pdf/2502.07603v3)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting does not consider general nonlinear dynamical systems. In developing this framework, we first consider the special case of linear driftless systems and recall the energies in the control signal in the nominal and malfunctioning systems. Using these energies, we derive a bound on the energetic resilience metric. For general nonlinear systems, we first obtain a condition on the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Assuming this approximation is exact, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A set of simulation examples demonstrate that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies for nonlinear systems.

摘要: 在本文中，我们通过研究由于致动器故障或对抗性攻击而部分失去控制权的系统的所有输入所使用的能量增加，来量化非线性动态系统的弹性。为了量化能量的最大增加，我们引入了能量弹性指标的概念。之前在这个特定环境下的工作没有考虑一般的非线性动力系统。在开发这个框架时，我们首先考虑线性无漂移系统的特殊情况，并回忆正常和故障系统中控制信号中的能量。使用这些能量，我们推导出能量弹性指标的界限。对于一般非线性系统，我们首先获得正常和故障系统中控制信号平均值的条件，这使我们能够逼近控制中的能量。然后，我们在所有故障输入上获得故障系统的该能量的最坏情况近似值。假设这种逼近是精确的，当一个致动器失去控制权时，我们推导出能量弹性指标的界限。一组模拟示例表明，尽管在获取非线性系统的控制能量时使用了近似值，但该指标对于量化系统的弹性是有用的，而无需显着的保守性。



## **5. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

NeurIPS 2025 (Spotlight)

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.15795v2) [paper-pdf](http://arxiv.org/pdf/2505.15795v2)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架（称为LLM as-a-Judge）具有高度可扩展性且成本相对较低。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。之前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给它们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了有关设计更可靠的法学硕士作为法官评估环境的重要问题。他们还证明，人类偏好可以通过管道化LLM来通过强化学习优化上游前级，从而有效地反向设计--这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **6. Dynamic Target Attack**

动态目标攻击 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.02422v2) [paper-pdf](http://arxiv.org/pdf/2510.02422v2)

**Authors**: Kedong Xiu, Churui Zeng, Tianhang Zheng, Xinzhe Huang, Xiaojun Jia, Di Wang, Puning Zhao, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks typically optimize an adversarial suffix to induce a fixed affirmative response. However, this fixed target usually resides in an extremely low-density region of a safety-aligned LLM's output distribution conditioned on diverse harmful inputs. Due to the substantial discrepancy between the target and the original output, existing attacks require numerous iterations to optimize the adversarial prompt, which might still fail to induce the low-probability target response from the target LLM. In this paper, we propose Dynamic Target Attack (DTA), a new jailbreaking framework relying on the target LLM's own responses as targets to optimize the adversarial prompts. In each optimization round, DTA iteratively samples multiple candidate responses directly from the output distribution conditioned on the current prompt, and selects the most harmful response as a temporary target for prompt optimization. In contrast to existing attacks, DTA significantly reduces the discrepancy between the target and the output distribution, substantially easing the optimization process to search for an effective adversarial prompt.   Extensive experiments demonstrate the superior effectiveness and efficiency of DTA: under the white-box setting, DTA only needs 200 optimization iterations to achieve an average attack success rate (ASR) of over 87\% on recent safety-aligned LLMs, exceeding the state-of-the-art baselines by over 15\%. The time cost of DTA is 2-26 times less than existing baselines. Under the black-box setting, DTA uses Llama-3-8B-Instruct as a surrogate model for target sampling and achieves an ASR of 85\% against the black-box target model Llama-3-70B-Instruct, exceeding its counterparts by over 25\%.

摘要: 现有的基于梯度的越狱攻击通常会优化一个对抗后缀来诱导一个固定的肯定响应。然而，该固定目标通常驻留在以各种有害输入为条件的安全对准LLM的输出分布的极低密度区域中。由于目标和原始输出之间存在很大差异，现有攻击需要多次迭代来优化对抗提示，这可能仍然无法从目标LLM中诱导出低概率的目标响应。在本文中，我们提出了动态目标攻击（DTA），一个新的越狱框架依赖于目标LLM自己的反应作为目标，以优化对抗性提示。在每轮优化中，DART直接从以当前提示为条件的输出分布中迭代采样多个候选响应，并选择最有害的响应作为即时优化的临时目标。与现有攻击相比，DART显着减少了目标和输出分布之间的差异，大大简化了搜索有效对抗提示的优化过程。   大量实验证明了DART的卓越有效性和效率：在白盒设置下，DART仅需要200次优化迭代，即可在最近的安全一致的LLM上实现超过87%的平均攻击成功率（ASB），超过最先进的基线15%以上。DART的时间成本比现有基线低2-26倍。在黑匣子设置下，DART使用Llama-3- 8B-Direct作为目标抽样的替代模型，与黑匣子目标模型Llama-3- 70 B-Direct的ASB相比，其ASB达到85%，超过同行25%以上。



## **7. AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches**

AngleRoCL：针对物理观点不变T2 I对抗补丁的角度稳健概念学习 cs.CV

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.09538v2) [paper-pdf](http://arxiv.org/pdf/2506.09538v2)

**Authors**: Wenjun Ji, Yuxiang Fu, Luyang Ying, Deng-Ping Fan, Yuyi Wang, Ming-Ming Cheng, Ivor Tsang, Qing Guo

**Abstract**: Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents. We released our code at https://github.com/tsingqguo/anglerocl.

摘要: 最前沿的作品表明，文本到图像（T2 I）扩散模型可以生成对抗补丁，误导物理世界中最先进的对象检测器，揭示检测器的漏洞和风险。然而，当从物理世界的不同角度观察时，这些方法忽视了T2 I补丁的攻击有效性（即，T2 I对抗补丁的角度稳健性）。本文全面研究了T2 I对抗补丁的角度鲁棒性，揭示了它们的角度鲁棒性问题，证明文本对生成补丁的角度鲁棒性有显着影响，而特定任务的语言指令未能增强角度鲁棒性。受这些研究的启发，我们引入了角度稳健概念学习（AngleRoCL），这是一种简单灵活的方法，可以学习可概括的概念（即，实现中的文本嵌入）表示生成角度稳健补丁的能力。学习到的概念可以被整合到文本提示中，并引导T2 I模型生成攻击有效性本质上可以抵抗观点变化的补丁。通过对多个视图的五个SOTA检测器进行广泛的模拟和物理世界实验，我们证明与基线方法相比，AngleRoCL显着增强了T2 I对抗斑块的角度稳健性。即使在具有挑战性的观看条件下，我们的补丁也能保持很高的攻击成功率，多个角度的攻击有效性平均相对提高超过50%。这项研究促进了对物理角度稳健补丁的理解，并深入了解T2 I生成的内容中的文本概念和物理属性之间的关系。我们在https://github.com/tsingqguo/anglerocl上发布了我们的代码。



## **8. Boosting Adversarial Transferability with Spatial Adversarial Alignment**

通过空间对抗对齐增强对抗可移植性 cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01015v2) [paper-pdf](http://arxiv.org/pdf/2501.01015v2)

**Authors**: Zhaoyu Chen, Haijing Guo, Kaixun Jiang, Jiyuan Fu, Xinyu Zhou, Dingkang Yang, Hao Tang, Bo Li, Wenqiang Zhang

**Abstract**: Deep neural networks are vulnerable to adversarial examples that exhibit transferability across various models. Numerous approaches are proposed to enhance the transferability of adversarial examples, including advanced optimization, data augmentation, and model modifications. However, these methods still show limited transferability, particularly in cross-architecture scenarios, such as from CNN to ViT. To achieve high transferability, we propose a technique termed Spatial Adversarial Alignment (SAA), which employs an alignment loss and leverages a witness model to fine-tune the surrogate model. Specifically, SAA consists of two key parts: spatial-aware alignment and adversarial-aware alignment. First, we minimize the divergences of features between the two models in both global and local regions, facilitating spatial alignment. Second, we introduce a self-adversarial strategy that leverages adversarial examples to impose further constraints, aligning features from an adversarial perspective. Through this alignment, the surrogate model is trained to concentrate on the common features extracted by the witness model. This facilitates adversarial attacks on these shared features, thereby yielding perturbations that exhibit enhanced transferability. Extensive experiments on various architectures on ImageNet show that aligned surrogate models based on SAA can provide higher transferable adversarial examples, especially in cross-architecture attacks.

摘要: 深度神经网络容易受到表现出跨各种模型可移植性的敌对示例的影响。人们提出了多种方法来增强对抗性示例的可移植性，包括高级优化、数据增强和模型修改。然而，这些方法仍然表现出有限的可移植性，特别是在跨体系结构场景中，例如从CNN到ViT。为了实现高可移植性，我们提出了一种名为空间对抗对齐（SBA）的技术，该技术利用对齐损失并利用见证模型来微调代理模型。具体来说，SBA由两个关键部分组成：空间感知对齐和对抗感知对齐。首先，我们最大限度地减少了全球和局部区域中两个模型之间的特征差异，以促进空间对齐。其次，我们引入了一种自我对抗策略，利用对抗示例来施加进一步的约束，从对抗的角度调整特征。通过这种对齐，代理模型被训练为专注于见证模型提取的共同特征。这促进了对这些共享特征的对抗攻击，从而产生表现出增强的可移植性的扰动。ImageNet上各种架构的广泛实验表明，基于SBA的对齐代理模型可以提供更高可转移的对抗性示例，尤其是在跨架构攻击中。



## **9. The Role of Information Incompleteness in Defending Against Stealth Attacks**

信息不完整在防御隐形攻击中的作用 eess.SY

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21227v1) [paper-pdf](http://arxiv.org/pdf/2510.21227v1)

**Authors**: Ke Sun, Jingyi Yan, Zhenglin Li, Shaorong Xie

**Abstract**: The effectiveness of Data Injections Attacks (DIAs) critically depends on the completeness of the system information accessible to adversaries. This relationship positions information incompleteness enhancement as a vital defense strategy for degrading DIA performance. In this paper, we focus on the information-theoretic stealth attacks, where the attacker encounters a fundamental tradeoff between the attack stealthiness and destructiveness. Specifically, we systematically characterize how incomplete admittance information impacts the dual objectives. In particular, we establish sufficient conditions for two distinct operational regimes: (i) stealthiness intensifies while destructive potential diminishes and (ii) destructiveness increases while stealth capability weakens. For scenarios beyond these regimes, we propose a maximal incompleteness strategy to optimally degrade stealth capability. To solve the associated optimization problem, the feasible region is reduced without excluding the optimal solution, and a heuristic algorithm is then introduced to effectively identify the near-optimal solutions within the reduced region. Numerical simulations are conducted on IEEE test systems to validate the findings.

摘要: 数据注入攻击（DIA）的有效性严重取决于对手可访问的系统信息的完整性。这种关系将信息不完整性增强定位为降低DIA性能的重要防御策略。本文重点研究信息论隐形攻击，攻击者在攻击隐形性和破坏性之间遇到了根本性的权衡。具体来说，我们系统地描述了不完整的准入信息如何影响双重目标。特别是，我们建立了两个不同的操作制度的充分条件：（一）隐形加强，而破坏性的潜力减少和（二）破坏性增加，而隐形能力减弱。对于超出这些制度的情况下，我们提出了一个最大的不完整性策略，以最佳地降低隐形能力。为了解决相关的优化问题，可行区域缩小而不排除最优解，然后引入启发式算法，以有效地识别缩小区域内的近优解。IEEE测试系统进行数值模拟，以验证研究结果。



## **10. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01741v2) [paper-pdf](http://arxiv.org/pdf/2501.01741v2)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是造成刻板印象和歧视的根深蒂固的手段。大型语言模型（LLM）现在是我们日常生活中一项普遍存在的技术，当容易产生有毒反应时，可能会造成广泛的伤害。解决这个问题的标准方法是调整LLM，然而，这会抑制这个问题，而不构成最终的解决方案。因此，即使在调整工作之后测试LLM对于检测道德标准的任何剩余偏差仍然至关重要。我们提出了EvoTox，一个自动化测试框架LLM的倾向毒性，提供了一种方法来定量评估有多少LLM可以推向毒性反应，即使在对齐的存在。该框架采用了一种迭代进化策略，利用两个LLM之间的相互作用，在测试系统（SUT）和提示发生器转向SUT响应更高的毒性。基于现有的毒性分类器，通过自动化oracle评估毒性水平。我们使用五个最先进的LLM作为评估对象进行定量和定性的实证评估，这些评估对象具有不断增加的复杂性（7- 671 B参数）。我们的定量评估根据现有基线方法评估了EvoTox的四种替代版本的成本效益，该方法基于随机搜索、精心策划的有毒提示数据集和对抗性攻击。我们的定性评估让人类评估人员对生成的提示的流畅性以及测试期间收集的反应的感知毒性进行评级。结果表明，就检测到的毒性水平而言，其有效性显着高于选定的基线方法（针对随机搜索的效果大小高达1.0，针对对抗性攻击的效果大小高达0.99）。此外，EvoTox的成本管理费用有限（平均从22%到35%）。



## **11. The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning**

木马示例：通过模板填充和不安全推理越狱LLM cs.CR

under review

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21190v1) [paper-pdf](http://arxiv.org/pdf/2510.21190v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok Yan Lam

**Abstract**: Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research.

摘要: 大型语言模型（LLM）发展迅速，现在编码了广泛的世界知识。然而，尽管进行了安全微调，它们仍然容易受到引发有害内容的对抗提示的影响。现有的越狱技术分为两类：白盒方法（例如，基于梯度的方法，例如GCG），需要模型内部结构，对于闭源API来说是不可行的，以及依赖攻击者LLM搜索或变异提示但通常产生缺乏可解释性和可移植性的模板的黑匣子方法。我们介绍TrojFill，一个黑盒越狱，将不安全的指令重构为模板填充任务。TrojFill嵌入混淆的有害指令（例如，通过占位符替换或Caesar/Base64编码），该模板要求模型（1）推理为什么原始指令是不安全的（不安全推理），以及（2）生成所请求文本的详细示例，然后进行逐句分析。关键的“示例”组件充当特洛伊木马，包含目标越狱内容，而周围的任务框架降低了拒绝率。我们根据领先LLM的标准越狱基准评估TrojFill（例如，ChatGPT、Gemini、DeepSeek、Qwen），表现出强劲的经验表现（例如，Gemini-Flash-2.5和DeepSeek-3.1上攻击成功率为100%，GPT-4 o上攻击成功率为97%）。此外，与先前的黑匣子优化方法相比，生成的提示表现出改进的可解释性和可移植性。我们发布代码、示例提示和生成的输出以支持未来的红色团队研究。



## **12. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.07736v3) [paper-pdf](http://arxiv.org/pdf/2506.07736v3)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **13. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

NeuGen Poisoning：通过外部知识的遗传优化对LLM检索增强生成的神经元引导攻击 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21144v1) [paper-pdf](http://arxiv.org/pdf/2510.21144v1)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够在推理期间动态集成外部知识，提高其事实准确性和适应性。然而，对手可以注入有毒的外部知识来覆盖模型的内部记忆。虽然现有的攻击迭代地操纵RAG的检索内容或提示结构，但它们在很大程度上忽略了模型的内部表示动态和神经元级敏感性。RAG中毒的根本机制尚未得到充分研究，也没有考虑RAG中知识冲突与强参数知识的影响。在这项工作中，我们提出了NeuGenPoisoning，这是一种新型攻击框架，可以在LLM内部神经元归因和遗传优化的指导下在RAG中生成对抗性外部知识。我们的方法首先识别出一组中毒反应神经元，其激活与上下文中毒知识密切相关。然后，我们采用遗传算法来进化对抗通道，最大限度地激活这些神经元。至关重要的是，我们的框架通过观察到的归因信号识别和重用有希望但最初不成功的外部知识变体，从而能够大规模地生成有效的有毒RAG知识。同时，中毒反应神经元引导的中毒可以有效地解决知识冲突。跨模型和数据集的实验结果表明，在保持流畅性的同时，始终实现了超过90%的高群体覆盖成功率（POSR）。实证结果表明，该方法有效地解决了知识冲突问题.



## **14. Alert-ME: An Explainability-Driven Defense Against Adversarial Examples in Transformer-Based Text Classification**

Alert-ME：基于转换器的文本分类中针对对抗性示例的解释性驱动防御 cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2307.01225v3) [paper-pdf](http://arxiv.org/pdf/2307.01225v3)

**Authors**: Bushra Sabir, Yansong Gao, Alsharif Abuadbba, M. Ali Babar

**Abstract**: Transformer-based text classifiers such as BERT, RoBERTa, T5, and GPT have shown strong performance in natural language processing tasks but remain vulnerable to adversarial examples. These vulnerabilities raise significant security concerns, as small input perturbations can cause severe misclassifications. Existing robustness methods often require heavy computation or lack interpretability. This paper presents a unified framework called Explainability-driven Detection, Identification, and Transformation (EDIT) to strengthen inference-time defenses. EDIT integrates explainability tools, including attention maps and integrated gradients, with frequency-based features to automatically detect and identify adversarial perturbations while offering insight into model behavior. After detection, EDIT refines adversarial inputs using an optimal transformation process that leverages pre-trained embeddings and model feedback to replace corrupted tokens. To enhance security assurance, EDIT incorporates automated alerting mechanisms that involve human analysts when necessary.   Beyond static defenses, EDIT also provides adaptive resilience by enforcing internal feature similarity and transforming inputs, thereby disrupting the attackers optimization process and limiting the effectiveness of adaptive adversarial attacks. Experiments using BERT and RoBERTa on IMDB, YELP, AGNEWS, and SST2 datasets against seven word substitution attacks demonstrate that EDIT achieves an average Fscore of 89.69 percent and balanced accuracy of 89.70 percent. Compared to four state-of-the-art defenses, EDIT improves balanced accuracy by 1.22 times and F1-score by 1.33 times while being 83 times faster in feature extraction. The framework provides robust, interpretable, and efficient protection against both standard, zero-day, and adaptive adversarial threats in text classification models.

摘要: BERT、RoBERTa、T5和GPT等基于转换器的文本分类器在自然语言处理任务中表现出出色的性能，但仍然容易受到对抗性示例的影响。这些漏洞引发了严重的安全问题，因为微小的输入扰动可能会导致严重的错误分类。现有的稳健性方法通常需要大量计算或缺乏可解释性。本文提出了一个名为解释性驱动的检测、识别和转换（EDIT）的统一框架，以加强推理时防御。EDIT集成了可解释性工具（包括注意力图和集成梯度）与基于频率的特征，以自动检测和识别对抗性扰动，同时提供对模型行为的洞察。检测后，EDIT使用最佳转换流程来细化对抗性输入，该流程利用预先训练的嵌入和模型反馈来替换损坏的令牌。为了增强安全保证，EDIT纳入了自动警报机制，必要时需要人力分析师参与。   除了静态防御之外，EDIT还通过强制执行内部特征相似性和转换输入来提供自适应弹性，从而扰乱攻击者的优化过程并限制自适应对抗攻击的有效性。在IMDB、YELP、AGNEWS和CST 2数据集上使用BERT和RoBERTa针对七个单词替换攻击的实验表明，EDIT的平均Fscore为89.69%，平衡准确率为89.70%。与四种最先进的防御相比，EDIT将平衡准确性提高了1.22倍，F1得分提高了1.33倍，同时特征提取速度提高了83倍。该框架提供了强大、可解释且高效的保护，以抵御文本分类模型中的标准、零日和自适应对抗威胁。



## **15. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations**

语言模型的元认知监控及其内部激活 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.13763v2) [paper-pdf](http://arxiv.org/pdf/2505.13763v2)

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).

摘要: 大型语言模型（LLM）有时可以报告它们实际用于解决任务的策略，但在其他时候似乎无法识别这些控制其行为的策略。这表明元认知程度有限--监控自己认知过程以进行后续报告和自我控制的能力。元认知增强了LLM解决复杂任务的能力，但也会引发安全问题，因为模型可能会混淆其内部流程以逃避基于神经激活的监督（例如，安全检测器）。鉴于社会对这些模型的依赖越来越大，我们了解它们的元认知能力至关重要。为了解决这个问题，我们引入了一种受神经科学启发的神经反馈范式，该范式使用上下文学习来量化LLM报告和控制其激活模式的元认知能力。我们证明它们的能力取决于几个因素：提供的上下文示例的数量、神经激活方向（要报告/控制）的语义解释性以及该方向解释的方差。这些方向跨越维度远低于模型神经空间的“元认知空间”，这表明LLM只能监控其神经激活的一小部分。我们的范式为量化LLM中的元认知提供了经验证据，对人工智能安全性具有重大影响（例如，对抗性攻击和防御）。



## **16. Can Current Detectors Catch Face-to-Voice Deepfake Attacks?**

当前的检测器可以捕捉面对面语音Deepfake攻击吗？ cs.CR

8 pages, Accepted at Workshop on AI for Cyber Threat Intelligence,  co-located with ACSAC 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.21004v1) [paper-pdf](http://arxiv.org/pdf/2510.21004v1)

**Authors**: Nguyen Linh Bao Nguyen, Alsharif Abuadbba, Kristen Moore, Tingming Wu

**Abstract**: The rapid advancement of generative models has enabled the creation of increasingly stealthy synthetic voices, commonly referred to as audio deepfakes. A recent technique, FOICE [USENIX'24], demonstrates a particularly alarming capability: generating a victim's voice from a single facial image, without requiring any voice sample. By exploiting correlations between facial and vocal features, FOICE produces synthetic voices realistic enough to bypass industry-standard authentication systems, including WeChat Voiceprint and Microsoft Azure. This raises serious security concerns, as facial images are far easier for adversaries to obtain than voice samples, dramatically lowering the barrier to large-scale attacks. In this work, we investigate two core research questions: (RQ1) can state-of-the-art audio deepfake detectors reliably detect FOICE-generated speech under clean and noisy conditions, and (RQ2) whether fine-tuning these detectors on FOICE data improves detection without overfitting, thereby preserving robustness to unseen voice generators such as SpeechT5.   Our study makes three contributions. First, we present the first systematic evaluation of FOICE detection, showing that leading detectors consistently fail under both standard and noisy conditions. Second, we introduce targeted fine-tuning strategies that capture FOICE-specific artifacts, yielding significant accuracy improvements. Third, we assess generalization after fine-tuning, revealing trade-offs between specialization to FOICE and robustness to unseen synthesis pipelines. These findings expose fundamental weaknesses in today's defenses and motivate new architectures and training protocols for next-generation audio deepfake detection.

摘要: 生成模型的快速发展使得人们能够创建越来越隐秘的合成语音，通常称为音频深度伪造。最近的一项技术FOICE [USENIX ' 24]展示了一种特别令人震惊的能力：从单个面部图像生成受害者的语音，而不需要任何语音样本。通过利用面部和声音特征之间的相关性，FOICE生成足够真实的合成语音，以绕过行业标准的认证系统，包括WeChat Voiceprint和Microsoft Azure。这引发了严重的安全问题，因为对手获取面部图像比获取语音样本更容易，从而大大降低了大规模攻击的障碍。在这项工作中，我们研究了两个核心研究问题：（PQ 1）最先进的音频深度伪造检测器能否在干净和有噪的条件下可靠地检测FOICE生成的语音，以及（PQ 2）根据FOICE数据微调这些检测器是否可以在不过度匹配的情况下改善检测，从而保留对SpeechT 5等不可见语音生成器的鲁棒性。   我们的研究做出了三点贡献。首先，我们对FOICE检测进行了首次系统评估，表明领先的检测器在标准和噪音条件下始终失败。其次，我们引入了有针对性的微调策略，以捕获FOICE特定的文物，从而显着提高准确性。第三，我们在微调后评估概括性，揭示FOICE的专业化和不可见合成管道的稳健性之间的权衡。这些发现暴露了当今防御的根本弱点，并激发了新的架构和训练协议用于下一代音频深度伪造检测。



## **17. Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference**

ATT&CK Insights的安全策略：利用LLM进行高级威胁理解和认知特征推断 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20930v1) [paper-pdf](http://arxiv.org/pdf/2510.20930v1)

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.   Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases

摘要: 理解网络安全中的对抗行为传统上依赖于高级情报报告和对攻击链的手动解释。然而，实时防御需要能够直接从入侵检测系统（IDS）日志等低级系统遥感数据中推断攻击者意图和认知策略。在本文中，我们提出了一种新颖的框架，该框架利用大型语言模型（LLM）来分析Suricata IDS日志并根据MITRE ATT & CK技术推断攻击者的行为。我们的方法基于这样的假设，即攻击者的行为反映了潜在的认知偏差，例如损失厌恶、风险容忍度或目标持续性，可以通过仔细观察日志序列来提取和建模。这为未来的行为适应性网络防御和认知特征推断工作奠定了基础。我们开发了一个策略驱动的提示系统，以高效的方式将大量网络日志数据细分为不同的行为阶段，使LLM能够将每个阶段与可能的技术和潜在的认知动机关联起来。通过将网络层事件映射到高级攻击者策略，我们的方法揭示了工具切换、协议转换或支点模式等行为信号如何对应于具有心理意义的决策点。结果表明，LLM可以弥合数据包级日志和战略意图之间的语义差距，为认知自适应网络防御提供途径。   关键词：认知网络安全、大型语言模型（LLM）、网络心理学、入侵检测系统（IDS）、MITRE ATT & CK、认知偏见



## **18. A new measure for dynamic leakage based on quantitative information flow**

基于量化信息流的动态泄漏新措施 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20922v1) [paper-pdf](http://arxiv.org/pdf/2510.20922v1)

**Authors**: Luigi D. C. Soares, Mário S. Alvim, Natasha Fernandes

**Abstract**: Quantitative information flow (QIF) is concerned with assessing the leakage of information in computational systems. In QIF there are two main perspectives for the quantification of leakage. On one hand, the static perspective considers all possible runs of the system in the computation of information flow, and is usually employed when preemptively deciding whether or not to run the system. On the other hand, the dynamic perspective considers only a specific, concrete run of the system that has been realised, while ignoring all other runs. The dynamic perspective is relevant for, e.g., system monitors and trackers, especially when deciding whether to continue or to abort a particular run based on how much leakage has occurred up to a certain point. Although the static perspective of leakage is well-developed in the literature, the dynamic perspective still lacks the same level of theoretical maturity. In this paper we take steps towards bridging this gap with the following key contributions: (i) we provide a novel definition of dynamic leakage that decouples the adversary's belief about the secret value from a baseline distribution on secrets against which the success of the attack is measured; (ii) we demonstrate that our formalisation satisfies relevant information-theoretic axioms, including non-interference and relaxed versions of monotonicity and the data-processing inequality (DPI); (iii) we identify under what kind of analysis strong versions of the axioms of monotonicity and the DPI might not hold, and explain the implications of this (perhaps counter-intuitive) outcome; (iv) we show that our definition of dynamic leakage is compatible with the well-established static perspective; and (v) we exemplify the use of our definition on the formalisation of attacks against privacy-preserving data releases.

摘要: 定量信息流（QIF）涉及评估计算系统中的信息泄漏。在QIF中，泄漏量化有两个主要角度。一方面，静态视角考虑了系统在信息流计算中的所有可能运行，并且通常在抢先决定是否运行系统时采用。另一方面，动态视角仅考虑已实现的系统的特定、具体运行，而忽略所有其他运行。动态视角与例如系统监视器和跟踪器，尤其是在根据直到某个点发生的泄漏量来决定是否继续或中止特定运行时。尽管静态泄漏视角在文献中得到了很好的发展，但动态视角仍然缺乏同等水平的理论成熟度。在本文中，我们采取措施通过以下关键贡献来弥合这一差距：（i）我们提供了动态泄露的一种新颖的定义，该定义将对手对秘密值的信念与衡量攻击成功的秘密的基线分布分开;（ii）我们证明我们的形式化满足相关的信息论公理，包括单调性和数据处理不平等（DPA）的非干扰和宽松版本;（iii）我们确定在什么样的分析下单调性公理和DPA的强版本可能不成立，并解释其含义（也许是违反直觉的）结果;（iv）我们表明我们对动态泄露的定义与公认的静态观点兼容;（v）我们重复使用我们的定义来形式化针对隐私保护数据发布的攻击。



## **19. Tex-ViT: A Generalizable, Robust, Texture-based dual-branch cross-attention deepfake detector**

Tex-ViT：一种可推广、稳健、基于纹理的双分支交叉注意深度伪造检测器 cs.CV

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2408.16892v2) [paper-pdf](http://arxiv.org/pdf/2408.16892v2)

**Authors**: Deepak Dagar, Dinesh Kumar Vishwakarma

**Abstract**: Deepfakes, which employ GAN to produce highly realistic facial modification, are widely regarded as the prevailing method. Traditional CNN have been able to identify bogus media, but they struggle to perform well on different datasets and are vulnerable to adversarial attacks due to their lack of robustness. Vision transformers have demonstrated potential in the realm of image classification problems, but they require enough training data. Motivated by these limitations, this publication introduces Tex-ViT (Texture-Vision Transformer), which enhances CNN features by combining ResNet with a vision transformer. The model combines traditional ResNet features with a texture module that operates in parallel on sections of ResNet before each down-sampling operation. The texture module then serves as an input to the dual branch of the cross-attention vision transformer. It specifically focuses on improving the global texture module, which extracts feature map correlation. Empirical analysis reveals that fake images exhibit smooth textures that do not remain consistent over long distances in manipulations. Experiments were performed on different categories of FF++, such as DF, f2f, FS, and NT, together with other types of GAN datasets in cross-domain scenarios. Furthermore, experiments also conducted on FF++, DFDCPreview, and Celeb-DF dataset underwent several post-processing situations, such as blurring, compression, and noise. The model surpassed the most advanced models in terms of generalization, achieving a 98% accuracy in cross-domain scenarios. This demonstrates its ability to learn the shared distinguishing textural characteristics in the manipulated samples. These experiments provide evidence that the proposed model is capable of being applied to various situations and is resistant to many post-processing procedures.

摘要: Deepfakes使用GAN来制作高度逼真的面部修饰，被广泛认为是流行的方法。传统的CNN已经能够识别虚假媒体，但它们很难在不同的数据集上表现良好，并且由于缺乏稳健性而容易受到对抗攻击。视觉转换器在图像分类问题领域已经展现出潜力，但它们需要足够的训练数据。受这些限制的启发，本出版物引入了Tex-ViT（纹理视觉Transformer），它通过将ResNet与视觉Transformer相结合来增强CNN功能。该模型将传统的ResNet特征与纹理模块相结合，该纹理模块在每次下采样操作之前对ResNet的部分进行并行操作。然后，纹理模块用作交叉注意视觉Transformer的双分支的输入。它特别注重改进提取特征地图相关性的全局纹理模块。经验分析表明，假图像表现出平滑的纹理，在操纵中在长距离内不会保持一致。在跨域场景中对不同类别的FF++（例如DF、f2 f、FS和NT）以及其他类型的GAN数据集进行了实验。此外，还对FF++、DFDCPreview和Celeb-DF数据集进行了实验，经历了多种后处理情况，例如模糊、压缩和噪音。该模型在概括性方面超越了最先进的模型，在跨领域场景中实现了98%的准确率。这证明了它能够学习处理样本中共同的区分纹理特征。这些实验提供了证据，表明所提出的模型能够应用于各种情况，并且能够抵抗许多后处理过程。



## **20. AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN**

AdaNOS：通过SDK中的深度对抗强化学习的自适应NOS攻击 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20566v1) [paper-pdf](http://arxiv.org/pdf/2510.20566v1)

**Authors**: Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe

**Abstract**: Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors.

摘要: 现有的防御机制在缓解基于规则的拒绝服务（DPS）攻击、利用预定义的签名和静态启发法来识别和阻止恶意流量方面表现出显着的有效性。然而，人工智能驱动技术的出现给SDK安全带来了新的挑战，可能会损害现有防御机制的有效性。本文中，我们介绍了~ AdaNOS，这是一种自适应攻击模型，可以通过对抗强化学习（RL）扰乱网络操作，同时逃避现有基于DoS的检测器的检测。具体来说，AdaNOS将该问题建模为攻击者和检测器之间的竞争游戏，攻击者的目标是在不被检测到的情况下阻止网络流量，而检测器的目标是识别恶意流量。AdaNOS可以根据来自dn和检测器的反馈动态调整其攻击策略来解决这个问题。此外，由于认识到攻击者通常比防御者拥有的信息少，AdaNOS类攻击将其定义为部分观察的马尔科夫决策过程（POMDP），攻击者只能访问攻击者和受害者节点之间的延迟信息。我们通过一个新颖的互惠学习模块来解决这一挑战，其中观察有限的学生代理通过向教师代理学习来提高其性能，教师代理在dn环境中具有完整的观察能力。AdaNOS代表了RL的第一个应用程序来开发类似于DoS的攻击序列，能够自适应地规避基于机器学习和基于规则的类似于DoS的攻击检测器。



## **21. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出非凡的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个关键问题出现了：当推理与危害交织在一起时，LRM是否会在推理模式中变得更容易越狱？为了研究这一点，我们引入了HauntAttack，这是一种新颖的通用黑匣子对抗攻击框架，它系统地将有害指令嵌入到推理问题中。具体来说，我们用有害指令修改现有问题中的关键推理条件，从而构建一条推理路径，引导模型逐步走向不安全的输出。我们对11种LRM进行了HauntAttack评估，观察到平均攻击成功率为70%，比之前最强的基线实现了高达12个百分点的绝对改进。我们的进一步分析表明，即使是先进的安全性一致的模型仍然极易受到基于推理的攻击，这为未来模型开发中平衡推理能力和安全性的紧迫挑战提供了见解。



## **22. Distributional Adversarial Attacks and Training in Deep Hedging**

分布式对抗攻击和深度对冲培训 math.OC

Camera-ready version (accepted at NeurIPS 2025  https://neurips.cc/virtual/2025/poster/115434)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2508.14757v2) [paper-pdf](http://arxiv.org/pdf/2508.14757v2)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Additional results indicate that the robust strategies maintain reliable performance on real market data and remain effective during periods of market change. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.

摘要: 本文通过利用对抗攻击的概念，研究了分布变化下经典深度对冲策略的稳健性。我们首先证明，标准的深度对冲模型极易受到输入分布中的微小扰动的影响，从而导致性能显着下降。受此启发，我们提出了一个对抗性训练框架，旨在提高深度对冲策略的稳健性。我们的方法将逐点对抗攻击扩展到分布式环境，并在Wasserstein球上引入了对抗优化问题的计算易于处理的重新公式。这使得能够有效地训练对分布扰动有弹性的对冲策略。通过广泛的数值实验，我们表明，在样本外性能和对模型错误指定的弹性方面，经过对抗训练的深度对冲策略始终优于经典对冲策略。其他结果表明，稳健的策略可以在真实市场数据上保持可靠的表现，并在市场变化期间保持有效。我们的研究结果为现实市场不确定性下的稳健深度对冲建立了一个实用有效的框架。



## **23. GUIDE: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models**

指南：通过去噪模型增强联邦学习中的梯度倒置攻击 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.17621v2) [paper-pdf](http://arxiv.org/pdf/2510.17621v2)

**Authors**: Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella, Mario Vento

**Abstract**: Federated Learning (FL) enables collaborative training of Machine Learning (ML) models across multiple clients while preserving their privacy. Rather than sharing raw data, federated clients transmit locally computed updates to train the global model. Although this paradigm should provide stronger privacy guarantees than centralized ML, client updates remain vulnerable to privacy leakage. Adversaries can exploit them to infer sensitive properties about the training data or even to reconstruct the original inputs via Gradient Inversion Attacks (GIAs). Under the honest-butcurious threat model, GIAs attempt to reconstruct training data by reversing intermediate updates using optimizationbased techniques. We observe that these approaches usually reconstruct noisy approximations of the original inputs, whose quality can be enhanced with specialized denoising models. This paper presents Gradient Update Inversion with DEnoising (GUIDE), a novel methodology that leverages diffusion models as denoising tools to improve image reconstruction attacks in FL. GUIDE can be integrated into any GIAs that exploits surrogate datasets, a widely adopted assumption in GIAs literature. We comprehensively evaluate our approach in two attack scenarios that use different FL algorithms, models, and datasets. Our results demonstrate that GUIDE integrates seamlessly with two state-ofthe- art GIAs, substantially improving reconstruction quality across multiple metrics. Specifically, GUIDE achieves up to 46% higher perceptual similarity, as measured by the DreamSim metric.

摘要: 联合学习（FL）支持跨多个客户端对机器学习（ML）模型进行协作训练，同时保护他们的隐私。联邦客户端不是共享原始数据，而是传输本地计算的更新来训练全球模型。尽管这种范式应该比集中式ML提供更强的隐私保证，但客户端更新仍然容易受到隐私泄露的影响。对手可以利用它们来推断训练数据的敏感属性，甚至通过梯度倒置攻击（GIA）重建原始输入。在诚实但好奇的威胁模型下，GIA试图通过使用基于优化的技术逆转中间更新来重建训练数据。我们观察到这些方法通常重建原始输入的有噪逼近，其质量可以通过专门的去噪模型来增强。本文介绍了带去噪的梯度更新倒置（GUIDE），这是一种新颖的方法，利用扩散模型作为去噪工具来改善FL中的图像重建攻击。GUIDE可以集成到任何利用代理数据集的GIA中，这是GIA文献中广泛采用的假设。我们在使用不同FL算法、模型和数据集的两种攻击场景中全面评估了我们的方法。我们的结果表明，GUIDE与两个最先进的GIA无缝集成，大幅提高了多个指标的重建质量。具体来说，通过DreamSim指标衡量，GUIDE的感知相似性提高了46%。



## **24. GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?**

GhostEEI-Bench：移动代理对动态设备上环境中的环境注入有弹性吗？ cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20333v1) [paper-pdf](http://arxiv.org/pdf/2510.20333v1)

**Authors**: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.

摘要: 视觉语言模型（VLM）越来越多地被部署为自治代理来导航移动图形用户界面（GUIs）。在动态的设备上生态系统（包括通知、弹出窗口和应用程序间交互）中运行，使它们面临一种独特且未充分探索的威胁载体：环境注入。与操纵文本指令的基于预算的攻击不同，环境注入通过将对抗性UI元素（例如，欺骗性覆盖或欺骗通知）直接插入到图形用户界面中来破坏代理的视觉感知。这绕过了文本保护措施，并可能会导致执行脱轨，导致隐私泄露、财务损失或不可逆转的设备损害。为了系统性地评估这种威胁，我们引入了GhostEI-Bench，这是第一个用于评估动态可执行环境中环境注入攻击下的移动代理的基准。GhostEEI-Bench超越了基于静态图像的评估，将对抗事件注入到完全运行的Android模拟器内的现实应用程序工作流程中，并评估关键风险场景中的性能。我们进一步提出了一种判断LLM协议，该协议通过审查代理的动作轨迹以及相应的屏幕截图序列来进行细粒度的失败分析，从而确定感知、识别或推理中的失败。对最先进代理的全面实验揭示了对欺骗性环境线索的明显脆弱性：当前的模型系统性地无法感知和推理被操纵的UI。GhostEEI-Bench提供了一个量化和缓解这种新出现的威胁的框架，为更强大、更安全的具体化代理铺平了道路。



## **25. Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses**

增强深度强化学习的安全性：对抗性攻击和防御的全面调查 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20314v1) [paper-pdf](http://arxiv.org/pdf/2510.20314v1)

**Authors**: Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun

**Abstract**: With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers.

摘要: 随着深度强化学习（DRL）技术在自动驾驶、智能制造、智能医疗等复杂领域的广泛应用，如何提高其在动态多变环境下的安全性和鲁棒性已成为当前研究的核心问题。特别是面对对抗攻击时，DRL可能会出现严重的性能下降，甚至做出潜在危险的决策，因此确保其在安全敏感场景中的稳定性至关重要。本文首先介绍了DRL的基本框架，并分析了复杂且不断变化的环境中面临的主要安全挑战。此外，本文提出了基于扰动类型和攻击目标的对抗性攻击分类框架，并详细回顾了针对DRL的主流对抗性攻击方法，包括扰动状态空间、动作空间、奖励函数和模型空间等各种攻击方法。为了有效对抗攻击，本文系统总结了当前各种鲁棒性训练策略，包括对抗性训练、竞争性训练、鲁棒学习、对抗性检测、防御提炼等相关防御技术，并讨论了这些方法在提高DRL鲁棒性方面的优点和缺点。最后，本文展望了对抗环境下DRL的未来研究方向，强调了在提高概括性、降低计算复杂性、增强可扩展性和可解释性方面的研究需求，旨在为研究人员提供有价值的参考和方向。



## **26. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

针对表格数据设计不可感知的Manifold对抗攻击 cs.LG

39 pages

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2507.10998v2) [paper-pdf](http://arxiv.org/pdf/2507.10998v2)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present unique challenges due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions. To address this, we propose a latent-space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate statistically consistent adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We introduce In-Distribution Success Rate (IDSR) to jointly evaluate attack effectiveness and distributional alignment. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches, achieving substantially lower outlier rates and higher IDSR across six datasets and three model architectures. Our comprehensive analyses of hyperparameter sensitivity, sparsity control, and generative architecture demonstrate that the effectiveness of VAE-based attacks depends strongly on reconstruction quality and the availability of sufficient training data. When these conditions are met, the proposed framework achieves superior practical utility and stability compared with input-space methods. This work underscores the importance of maintaining on-manifold perturbations for generating realistic and robust adversarial examples in tabular domains.

摘要: 由于混合类别和数字特征的多样性，对表格数据的对抗攻击带来了独特的挑战。与像素扰动保持视觉相似性的图像不同，表格数据缺乏直观的相似性指标，因此很难定义难以察觉的修改。此外，传统的基于梯度的方法优先考虑$\ell_p$-norm约束，通常会产生偏离原始数据分布的对抗性示例。为了解决这个问题，我们提出了一种潜空间扰动框架，使用混合输入变分自动编码器（VAE）来生成统计上一致的对抗示例。提出的VAE将类别嵌入和数字特征集成到统一的潜在多管齐中，从而实现保持统计一致性的扰动。我们引入分布式成功率（IDSR）来联合评估攻击有效性和分布式对齐。对六个公开可用数据集和三个模型架构的评估表明，与传统的输入空间攻击和从图像域方法改编的其他基于VAE的方法相比，我们的方法实现了显着更低的异常值率和更一致的性能，实现了显着更低的异常值率和更高的IDSR跨六个数据集和三个模型架构。我们对超参数敏感性、稀疏性控制和生成式架构的全面分析表明，基于VAE的攻击的有效性在很大程度上取决于重建质量和足够训练数据的可用性。当满足这些条件时，与输入空间方法相比，所提出的框架实现了更好的实际实用性和稳定性。这项工作强调了维持管汇上扰动对于在表格域中生成现实且稳健的对抗示例的重要性。



## **27. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

超越文本：通过感性简单的转换对视觉语言和音频模型进行多模式越狱 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.

摘要: 多模式大型语言模型（MLLM）已经取得了显着的进展，但仍然极易受到利用跨模式处理弱点的对抗攻击的影响。我们对针对视觉语言和音频语言模型的多模式越狱进行了系统性研究，表明即使是简单的感知转换也可以可靠地绕过最先进的安全过滤器。我们的评估涵盖了三个高风险安全类别有害内容、CBRN（化学、生物、放射、核）和CTEM（儿童性剥削材料）的1，900个对抗性提示，针对七个前沿模型进行了测试。我们探索了MLLM攻击技术的有效性，包括FigStep-Pro（视觉关键字分解）、智能掩蔽（语义混淆）和音频扰动（Wave-Echo、Wave-Pitch、Wave-Speed）。结果揭示了严重的漏洞：在感知修改的输入下，具有几乎完美的纯文本安全性（0\%ASB）的模型遭受了超过75%的攻击成功率，而FigStep-Pro在Lama-4变体中实现了高达89%的ASB。基于音频的攻击进一步揭示了提供商特定的弱点，即使是基本的模式传输也会产生25%的技术查询的ASB。这些发现暴露了以文本为中心的对齐和多模式威胁之间的关键差距，表明当前的保障措施未能普遍适用于跨模式攻击。这些攻击的可访问性需要最少的技术专业知识，这表明强大的多模式人工智能安全性需要范式转向更广泛的语义层面推理，以减轻可能的风险。



## **28. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

TRUST：审计大型语言模型推理的去中心化框架 cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.

摘要: 大型语言模型生成复杂的推理链，揭示其决策，但验证这些中间步骤的忠实性和无害性仍然是一个尚未解决的关键问题。现有的审计方法集中、不透明且难以扩展，为在高风险领域部署专有模型带来了巨大风险。我们确定了四个核心挑战：（1）稳健性：集中式审计员是单点失败，容易受到偏见或攻击。(2)可扩展性：推理轨迹太长，无法手动验证。(3)不透明：封闭审计破坏了公众信任。(4)隐私：暴露完整推理可能会导致模型被盗或提炼。我们提出TRUST，这是一个透明的、去中心化的审计框架，通过以下方式克服这些限制：（1）不同审计员之间的共识机制，保证在高达30%的恶意参与者下的正确性。(2)推理痕迹的分层DAB分解，实现可扩展的并行审计。(3)一个区块链分类帐，记录所有验证决定，以供公众问责。(4)保留隐私的分段，仅共享部分推理步骤以保护专有逻辑。我们为TRUST框架的安全性和经济激励提供理论保证。跨多个LLM（GPT-OSS、DeepSeek-r1、Qwen）和推理任务（数学、医学、科学、人文学科）的实验表明，TRUST有效地检测推理缺陷，并在对抗性审计员的情况下保持稳健性。我们的工作开创了去中心化的人工智能审计，为安全且值得信赖的LLM部署提供了实用途径。



## **29. Active Localization of Close-range Adversarial Acoustic Sources for Underwater Data Center Surveillance**

水下数据中心监控中近距离对抗声源的主动定位 eess.SP

12 pages, V1

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20122v1) [paper-pdf](http://arxiv.org/pdf/2510.20122v1)

**Authors**: Adnan Abdullah, David Blow, Sara Rampazzi, Md Jahidul Islam

**Abstract**: Underwater data infrastructures offer natural cooling and enhanced physical security compared to terrestrial facilities, but are susceptible to acoustic injection attacks that can disrupt data integrity and availability. This work presents a comprehensive surveillance framework for localizing and tracking close-range adversarial acoustic sources targeting offshore infrastructures, particularly underwater data centers (UDCs). We propose a heterogeneous receiver configuration comprising a fixed hydrophone mounted on the facility and a mobile hydrophone deployed on a dedicated surveillance robot. While using enough arrays of static hydrophones covering large infrastructures is not feasible in practice, off-the-shelf approaches based on time difference of arrival (TDOA) and frequency difference of arrival (FDOA) filtering fail to generalize for this dynamic configuration. To address this, we formulate a Locus-Conditioned Maximum A-Posteriori (LC-MAP) scheme to generate acoustically informed and geometrically consistent priors, ensuring a physically plausible initial state for a joint TDOA-FDOA filtering. We integrate this into an unscented Kalman filtering (UKF) pipeline, which provides reliable convergence under nonlinearity and measurement noise. Extensive Monte Carlo analyses, Gazebo-based physics simulations, and field trials demonstrate that the proposed framework can reliably estimate the 3D position and velocity of an adversarial acoustic attack source in real time. It achieves sub-meter localization accuracy and over 90% success rates, with convergence times nearly halved compared to baseline methods. Overall, this study establishes a geometry-aware, real-time approach for acoustic threat localization, advancing autonomous surveillance capabilities of underwater infrastructures.

摘要: 与陆地设施相比，水下数据基础设施提供自然冷却和增强的物理安全性，但很容易受到声学注入攻击，从而破坏数据的完整性和可用性。这项工作提供了一个全面的监视框架，用于定位和跟踪针对海上基础设施（特别是水下数据中心（UDC））的近距离对抗性声学源。我们提出了一种异类接收器配置，包括安装在设施上的固定式水下听音器和部署在专用监视机器人上的移动式水下听音器。虽然在实践中使用足够多的静态声纳阵列覆盖大型基础设施是不可行的，但基于到达时间差（TDOE）和到达频率差（FDOA）过滤的现成方法无法普遍适用于这种动态配置。为了解决这个问题，我们制定了一种定位条件最大A后验（LC-MAP）方案来生成声学信息和几何一致的先验，确保联合TDOA-FDOA过滤的物理上合理的初始状态。我们将其集成到无踪迹卡尔曼过滤（UKF）管道中，该管道在非线性和测量噪音下提供可靠的收敛。广泛的蒙特卡洛分析、基于Gazebo的物理模拟和现场试验表明，所提出的框架可以可靠地实时估计对抗性声学攻击源的3D位置和速度。它实现了亚米定位准确度和超过90%的成功率，与基线方法相比，收敛时间几乎减少了一半。总体而言，这项研究建立了一种几何感知的实时声学威胁定位方法，提高水下基础设施的自主监视能力。



## **30. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **31. Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness**

弥合对称性和鲁棒性：论等方差在增强对抗鲁棒性中的作用 cs.LG

Accepted for the proceedings of 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.16171v2) [paper-pdf](http://arxiv.org/pdf/2510.16171v2)

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh, Chaowei Zhang, Xiao Qin, Yang Zhou

**Abstract**: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation- and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.

摘要: 对抗性示例通过利用深度神经网络对不可感知的输入扰动的敏感性来揭示深度神经网络中的关键漏洞。虽然对抗训练仍然是主要的防御策略，但它通常会产生巨大的计算成本，并可能会损害干净数据的准确性。在这项工作中，我们研究了一种对抗鲁棒性的架构方法，通过将组等变卷积（具体来说是旋转和规模等变层）嵌入到标准卷积神经网络（CNN）中。这些层编码对称先验，将模型行为与输入空间中的结构化转换保持一致，从而促进更平滑的决策边界和更大的对抗性攻击弹性。我们提出并评估了两种混合感知架构：在融合之前独立处理标准和等变特征的并行设计，以及顺序应用等变操作的级联设计。从理论上讲，我们证明此类模型可以降低假设空间的复杂性、规范化梯度，并在CIVER（nEtwork稳健性的Cross Lipschitz Extreme Value）框架下产生更严格的认证稳健性界限。从经验上看，我们的模型在FGSM和PVD攻击下一致提高了CIFAR-10、CIFAR-100和CIFAR-10 C的对抗稳健性和概括性，而无需对抗训练。这些发现强调了安全执行架构作为基于数据增强的防御的高效且有原则的替代方案的潜力。



## **32. JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation**

JaiLIP：通过损失引导图像扰动越狱的视觉语言模型 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21401v2) [paper-pdf](http://arxiv.org/pdf/2509.21401v2)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Vision-Language Models (VLMs) have remarkable abilities in generating multimodal reasoning tasks. However, potential misuse or safety alignment concerns of VLMs have increased significantly due to different categories of attack vectors. Among various attack vectors, recent studies have demonstrated that image-based perturbations are particularly effective in generating harmful outputs. In the literature, many existing techniques have been proposed to jailbreak VLMs, leading to unstable performance and visible perturbations. In this study, we propose Jailbreaking with Loss-guided Image Perturbation (JaiLIP), a jailbreaking attack in the image space that minimizes a joint objective combining the mean squared error (MSE) loss between clean and adversarial image with the models harmful-output loss. We evaluate our proposed method on VLMs using standard toxicity metrics from Perspective API and Detoxify. Experimental results demonstrate that our method generates highly effective and imperceptible adversarial images, outperforming existing methods in producing toxicity. Moreover, we have evaluated our method in the transportation domain to demonstrate the attacks practicality beyond toxic text generation in specific domain. Our findings emphasize the practical challenges of image-based jailbreak attacks and the need for efficient defense mechanisms for VLMs.

摘要: 视觉语言模型（VLM）在生成多模式推理任务方面具有非凡的能力。然而，由于攻击载体类型的不同，VLM的潜在误用或安全对齐问题显着增加。在各种攻击载体中，最近的研究表明，基于图像的扰动在产生有害输出方面特别有效。在文献中，许多现有技术都被提出来越狱VLM，导致性能不稳定和可见的扰动。在这项研究中，我们提出了具有损失引导图像扰动的越狱（JaiLIP），这是图像空间中的一种越狱攻击，可最大限度地减少将干净图像和对抗图像之间的均方误差（SSE）损失与模型有害输出损失相结合的联合目标。我们使用Perspective API和Deepfy的标准毒性指标评估我们在VLM上提出的方法。实验结果表明，我们的方法可以生成高效且难以感知的对抗图像，在产生毒性方面优于现有方法。此外，我们在交通领域评估了我们的方法，以证明除了特定领域有毒文本生成之外的攻击的实用性。我们的研究结果强调了基于图像的越狱攻击的实际挑战以及对VLM有效防御机制的需求。



## **33. Sharp Gaussian approximations for Decentralized Federated Learning**

分散式联邦学习的尖锐高斯逼近 stat.ML

Accepted as Spotlight, NeurIPS'25, Main Conference Track

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.08125v2) [paper-pdf](http://arxiv.org/pdf/2505.08125v2)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.

摘要: 联邦学习在隐私敏感的协作环境中获得了吸引力，本地新元正在成为去中心化环境中的关键优化方法。虽然它的收敛性质已得到充分研究，但超越收敛的渐进统计保证仍然有限。在本文中，我们给出了局部SGD的两个广义高斯逼近结果，并探讨了它们的含义。首先，我们证明了最终本地BCD迭代的Berry-Esseen定理，从而实现有效的乘数引导过程。其次，出于鲁棒性考虑，我们为局部SGD的整个轨迹引入了两种不同的时间均匀高斯逼近。时间一致的逼近支持基于高斯引导的测试来检测对抗性攻击。提供了广泛的模拟来支持我们的理论结果。



## **34. QORE : Quantum Secure 5G/B5G Core**

QORE：Quantum安全5G/B5 G核心 cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19982v1) [paper-pdf](http://arxiv.org/pdf/2510.19982v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Rudraksh Rawal, Nitin Rajput, Shiva Valia, Madhav Aggarwal, Aditya Gairola

**Abstract**: Quantum computing is reshaping the security landscape of modern telecommunications. The cryptographic foundations that secure todays 5G systems, including RSA, Elliptic Curve Cryptography (ECC), and Diffie-Hellman (DH), are all susceptible to attacks enabled by Shors algorithm. Protecting 5G networks against future quantum adversaries has therefore become an urgent engineering and research priority. In this paper we introduce QORE, a quantum-secure 5G and Beyond 5G (B5G) Core framework that provides a clear pathway for transitioning both the 5G Core Network Functions and User Equipment (UE) to Post-Quantum Cryptography (PQC). The framework uses the NIST-standardized lattice-based algorithms Module-Lattice Key Encapsulation Mechanism (ML-KEM) and Module-Lattice Digital Signature Algorithm (ML-DSA) and applies them across the 5G Service-Based Architecture (SBA). A Hybrid PQC (HPQC) configuration is also proposed, combining classical and quantum-safe primitives to maintain interoperability during migration. Experimental validation shows that ML-KEM achieves quantum security with minor performance overhead, meeting the low-latency and high-throughput requirements of carrier-grade 5G systems. The proposed roadmap aligns with ongoing 3GPP SA3 and SA5 study activities on the security and management of post-quantum networks as well as with NIST PQC standardization efforts, providing practical guidance for mitigating quantum-era risks while safeguarding long-term confidentiality and integrity of network data.

摘要: 量子计算正在重塑现代电信的安全格局。保护当今5G系统安全的加密基础，包括RSA、椭圆曲线密码学（EC）和迪夫-赫尔曼（DH），都容易受到Shors算法发起的攻击。因此，保护5G网络免受未来量子对手的侵害已成为紧迫的工程和研究优先事项。在本文中，我们介绍了QORE，这是一个量子安全的5G和超越5G（B5 G）核心框架，它为将5G核心网络功能和用户设备（UE）过渡到后量子加密（PQC）提供了明确的途径。该框架使用NIST标准化的基于格的算法模块格密钥封装机制（ML-KEM）和模块格数字签名算法（ML-DSA），并将其应用于5G基于服务的架构（SBA）。还提出了一种混合PQC（HPQC）配置，结合经典和量子安全原语，以保持迁移过程中的互操作性。实验验证表明，ML-KEM以较小的性能开销实现了量子安全，满足了运营商级5G系统的低延迟和高吞吐量要求。拟议的路线图与正在进行的3GPP SA 3和SA 5关于后量子网络安全和管理的研究活动以及NIST PQC标准化工作保持一致，为减轻量子时代的风险提供实用指导，同时保护网络数据的长期机密性和完整性。



## **35. Towards Strong Certified Defense with Universal Asymmetric Randomization**

通过普遍不对称随机化实现强大的认证防御 cs.LG

Accepted by CSF 2026, 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19977v1) [paper-pdf](http://arxiv.org/pdf/2510.19977v1)

**Authors**: Hanbin Hong, Ashish Kundu, Ali Payani, Binghui Wang, Yuan Hong

**Abstract**: Randomized smoothing has become essential for achieving certified adversarial robustness in machine learning models. However, current methods primarily use isotropic noise distributions that are uniform across all data dimensions, such as image pixels, limiting the effectiveness of robustness certification by ignoring the heterogeneity of inputs and data dimensions. To address this limitation, we propose UCAN: a novel technique that \underline{U}niversally \underline{C}ertifies adversarial robustness with \underline{A}nisotropic \underline{N}oise. UCAN is designed to enhance any existing randomized smoothing method, transforming it from symmetric (isotropic) to asymmetric (anisotropic) noise distributions, thereby offering a more tailored defense against adversarial attacks. Our theoretical framework is versatile, supporting a wide array of noise distributions for certified robustness in different $\ell_p$-norms and applicable to any arbitrary classifier by guaranteeing the classifier's prediction over perturbed inputs with provable robustness bounds through tailored noise injection. Additionally, we develop a novel framework equipped with three exemplary noise parameter generators (NPGs) to optimally fine-tune the anisotropic noise parameters for different data dimensions, allowing for pursuing different levels of robustness enhancements in practice.Empirical evaluations underscore the significant leap in UCAN's performance over existing state-of-the-art methods, demonstrating up to $182.6\%$ improvement in certified accuracy at large certified radii on MNIST, CIFAR10, and ImageNet datasets.\footnote{Code is anonymously available at \href{https://github.com/youbin2014/UCAN/}{https://github.com/youbin2014/UCAN/}}

摘要: 随机平滑对于在机器学习模型中实现经过认证的对抗鲁棒性至关重要。然而，当前的方法主要使用在所有数据维度（例如图像像素）上均匀的各向同性噪音分布，从而通过忽略输入和数据维度的同质性来限制稳健性认证的有效性。为了解决这一限制，我们提出了UCAN：一种新型技术，通过\underworldly\underworldly\underworldly {C}通过\underworlded {A} nentropic\underworlded {N}oise来证明对抗鲁棒性。UCAN旨在增强任何现有的随机平滑方法，将其从对称（各向同性）噪音分布转换为不对称（各向异性）噪音分布，从而提供更定制的防御对抗攻击。我们的理论框架是通用的，支持广泛的噪音分布，以在不同的$\ell_p$-规范中获得认证的鲁棒性，并且通过定制的噪音注入保证分类器对具有可证明的鲁棒性边界的扰动输入的预测，适用于任何任意分类器。此外，我们开发了一个配备三个示例性噪音参数发生器（NPG）的新型框架，以最佳方式微调不同数据维度的各向异性噪音参数，从而在实践中追求不同水平的鲁棒性增强。经验评估强调了UCAN性能相对于现有最先进方法的显着飞跃，在MNIST、CIFAR 10和ImageNet数据集上，在大认证半径下，认证准确性提高了高达182.6美元。\脚注{代码可在\href{https：//github.com/youbin2014/UCAN/}{https：//github.com/youbin2014/UCAN/}}



## **36. Q-RAN: Quantum-Resilient O-RAN Architecture**

Q-RAN：量子弹性O-RAN架构 cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19968v1) [paper-pdf](http://arxiv.org/pdf/2510.19968v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Madhav Agarwal, Nitin Rajput, Kriish Sharma, Sushant Mundepi, Shivam Gangwar, Rudraksh Rawal, Jishan

**Abstract**: The telecommunications industry faces a dual transformation: the architectural shift toward Open Radio Access Networks (O-RAN) and the emerging threat from quantum computing. O-RAN disaggregated, multi-vendor architecture creates a larger attack surface vulnerable to crypt-analytically relevant quantum computers(CRQCs) that will break current public key cryptography. The Harvest Now, Decrypt Later (HNDL) attack strategy makes this threat immediate, as adversaries can intercept encrypted data today for future decryption. This paper presents Q-RAN, a comprehensive quantum-resistant security framework for O-RAN networks using NIST-standardized Post-Quantum Cryptography (PQC). We detail the implementation of ML-KEM (FIPS 203) and ML-DSA (FIPS 204), integrated with Quantum Random Number Generators (QRNG) for cryptographic entropy. The solution deploys PQ-IPsec, PQ-DTLS, and PQ-mTLS protocols across all O-RAN interfaces, anchored by a centralized Post-Quantum Certificate Authority (PQ-CA) within the SMO framework. This work provides a complete roadmap for securing disaggregated O-RAN ecosystems against quantum adversaries.

摘要: 电信行业面临双重转型：向开放式无线电接入网络（O-RAN）的架构转变以及来自量子计算的新威胁。O-RAN分解的多供应商架构创建了更大的攻击面，容易受到密码分析相关量子计算机（CRQC）的攻击，这将打破当前的公钥加密技术。立即收获，稍后解密（HNDL）攻击策略使这种威胁迫在眉睫，因为对手可以立即拦截加密数据，以便将来解密。本文介绍了Q-RAN，这是一个全面的抗量子安全框架，用于O-RAN网络，使用NIH标准化的后量子密码学（PQC）。我们详细介绍了ML-KEM（TIP 203）和ML-DSA（TIP 204）的实现，并与量子随机数发生器（QRNG）集成以实现加密信息的信息。该解决方案跨所有O-RAN接口部署PQ-SYS、PQ-DTLS和PQ-mSSL协议，并由NSO框架内的集中式后量子证书颁发机构（PQ-CA）锚定。这项工作为保护分散的O-RAN生态系统免受量子对手的侵害提供了完整的路线图。



## **37. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

未学习但未被遗忘：LLM中精确未学习后的数据提取 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。



## **38. Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?**

现代语音增强系统容易受到对抗攻击吗？ eess.AS

Copyright 2026 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21087v2) [paper-pdf](http://arxiv.org/pdf/2509.21087v2)

**Authors**: Rostislav Makarov, Lea Schönherr, Timo Gerkmann

**Abstract**: Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.

摘要: 用于语音增强的机器学习方法正变得越来越具有表达力，从而能够对输入信号进行更强大的修改。在本文中，我们证明了这种表现力引入了一个漏洞：高级语音增强模型可能容易受到对抗性攻击。具体来说，我们表明，可以注入经过精心设计并由原始输入在心理声学上掩盖的对抗性噪音，以便增强的语音输出传达完全不同的语义含义。我们通过实验验证了当代预测语音增强模型确实可以以这种方式操纵。此外，我们强调，具有随机采样器的扩散模型通过设计表现出对此类对抗性攻击的固有鲁棒性。



## **39. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

关于在异类网络中扩展LT编码区块链及其对拒绝服务威胁的脆弱性 cs.IT

To appear in Future Generation Computer Systems, 2025. This is an  extended version of a shorter version that has appeared in IEEE ICC 2024

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2402.05620v3) [paper-pdf](http://arxiv.org/pdf/2402.05620v3)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.

摘要: 编码区块链作为降低存储成本和促进可扩展性的有希望的解决方案而受到重视。在这一类中，卢比变换（LT）编码区块链是一个有吸引力的可扩展性选择，因为它可以提供广泛的低复杂度解码器。在本工作的第一部分中，我们发现传统的LT解码器（如Belief Propagation和On-the-Fly Gaussian Elimation）对于节点具有不同计算和下载能力的异类网络来说可能不是最佳选择。为了解决这个问题，我们引入了一系列LT代码的混合解码器，并为它们提出了最佳操作机制，以便以最低的解码成本恢复区块链。虽然人们从存储节省和可扩展性方面研究了LT编码的区块链架构，但对其安全漏洞知之甚少。针对这一研究空白，在第二部分中，我们在LT编码区块链上提出了新型的拒绝服务威胁，这些威胁针对具有特定解码能力的节点，阻止它们加入网络。我们提出的威胁本质上是非无意识的，其中对手可以访问存档的块，并根据底层编码方案选择对其中的子集执行攻击。我们表明，我们的优化威胁可以在有限的资源下实现与盲目攻击相同程度的破坏。总的来说，这是第一个此类工作，它提出了设计编码区块链的新问题，以共同提供存储节省、可扩展性以及对优化威胁的弹性。



## **40. Exploring the Effect of DNN Depth on Adversarial Attacks in Network Intrusion Detection Systems**

探讨DNN深度对网络入侵检测系统中对抗攻击的影响 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19761v1) [paper-pdf](http://arxiv.org/pdf/2510.19761v1)

**Authors**: Mohamed ElShehaby, Ashraf Matrawy

**Abstract**: Adversarial attacks pose significant challenges to Machine Learning (ML) systems and especially Deep Neural Networks (DNNs) by subtly manipulating inputs to induce incorrect predictions. This paper investigates whether increasing the layer depth of deep neural networks affects their robustness against adversarial attacks in the Network Intrusion Detection System (NIDS) domain. We compare the adversarial robustness of various deep neural networks across both \ac{NIDS} and computer vision domains (the latter being widely used in adversarial attack experiments). Our experimental results reveal that in the NIDS domain, adding more layers does not necessarily improve their performance, yet it may actually significantly degrade their robustness against adversarial attacks. Conversely, in the computer vision domain, adding more layers exhibits a more modest impact on robustness. These findings can guide the development of robust neural networks for (NIDS) applications and highlight the unique characteristics of network security domains within the (ML) landscape.

摘要: 对抗性攻击通过巧妙地操纵输入来引发错误的预测，对机器学习（ML）系统，尤其是深度神经网络（DNN）构成了重大挑战。本文研究了增加深度神经网络的层深度是否会影响其在网络入侵检测系统（NIDS）领域中对抗性攻击的鲁棒性。我们比较了\ac{NIDS}和计算机视觉领域（后者被广泛用于对抗攻击实验）的各种深度神经网络的对抗鲁棒性。我们的实验结果表明，在NIDS领域，添加更多层并不一定会提高它们的性能，但实际上可能会显着降低它们对对抗攻击的鲁棒性。相反，在计算机视觉领域，添加更多层对稳健性的影响更为温和。这些发现可以指导（NIDS）应用程序的稳健神经网络的开发，并强调（ML）环境中网络安全域的独特特征。



## **41. Explainable Face Presentation Attack Detection via Ensemble-CAM**

通过Ensemble-CAM检测可解释的面部呈现攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19695v1) [paper-pdf](http://arxiv.org/pdf/2510.19695v1)

**Authors**: Rashik Shadman, M G Sarwar Murshed, Faraz Hussain

**Abstract**: Presentation attacks represent a critical security threat where adversaries use fake biometric data, such as face, fingerprint, or iris images, to gain unauthorized access to protected systems. Various presentation attack detection (PAD) systems have been designed leveraging deep learning (DL) models to mitigate this type of threat. Despite their effectiveness, most of the DL models function as black boxes - their decisions are opaque to their users. The purpose of explainability techniques is to provide detailed information about the reason behind the behavior or decision of DL models. In particular, visual explanation is necessary to better understand the decisions or predictions of DL-based PAD systems and determine the key regions due to which a biometric image is considered real or fake by the system. In this work, a novel technique, Ensemble-CAM, is proposed for providing visual explanations for the decisions made by deep learning-based face PAD systems. Our goal is to improve DL-based face PAD systems by providing a better understanding of their behavior. Our provided visual explanations will enhance the transparency and trustworthiness of DL-based face PAD systems.

摘要: 演示攻击是一种严重的安全威胁，对手使用虚假生物识别数据（例如面部、指纹或虹膜图像）来未经授权地访问受保护的系统。已经设计了各种演示攻击检测（PAD）系统，利用深度学习（DL）模型来缓解此类威胁。尽管它们有效，但大多数DL模型都像黑匣子一样发挥作用--它们的决定对用户来说是不透明的。可解释性技术的目的是提供有关DL模型行为或决策背后原因的详细信息。特别是，视觉解释对于更好地理解基于DL的PAD系统的决策或预测并确定系统认为生物识别图像真实或虚假的关键区域是必要的。在这项工作中，提出了一种新颖的技术Ensemble-CAM，用于为基于深度学习的面部DPP系统做出的决策提供视觉解释。我们的目标是通过更好地了解基于DL的面部DPP系统的行为来改进其。我们提供的视觉解释将增强基于DL的面部DPP系统的透明度和可信度。



## **42. Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent**

风格攻击伪装：当字体成为对抗意图的伪装 cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19641v1) [paper-pdf](http://arxiv.org/pdf/2510.19641v1)

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation.

摘要: 随着社交媒体的发展，用户使用风格字体和类似字体的表情符号来表达个性，创建视觉吸引力且保持人类可读的文本。然而，这些字体在NLP模型中引入了隐藏的漏洞：虽然人类很容易阅读风格文本，但模型将这些字符作为不同的标记处理，从而造成干扰。我们识别了这种人类模型的感知差距，并提出了一种基于风格的攻击，即风格攻击伪装（SAD）。我们设计了两种尺寸：轻型用于查询效率，重型用于卓越的攻击性能。跨传统模型、LLM和商业服务的情感分类和机器翻译实验证明了SAD的强大攻击性能。我们还展示了SAD对多模式任务（包括文本到图像和文本到语音生成）的潜在威胁。



## **43. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

你能相信你所看到的吗？Alpha通道对视频对象检测的无框攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.

摘要: 随着对象检测模型越来越多地部署在自动驾驶汽车（AV）和监控平台等网络物理系统中，确保其针对对抗威胁的安全性至关重要。虽然之前的工作探讨了图像领域中的对抗攻击，但视频领域中的这些攻击在很大程度上仍然没有得到审查，尤其是在无框环境中。在本文中，我们介绍了{\Alpha}-Cloak，这是对对象检测器的第一个无箱对抗攻击，完全通过RGBA视频的Alpha通道进行操作。{\Alpha}-Cloak利用Alpha通道将恶意目标视频与良性视频融合，导致融合后的视频对人类观看者来说似乎无害，但始终欺骗对象检测器。我们的攻击不需要访问模型架构、参数或输出，并且不会引入可感知的伪影。我们系统性地研究了对常见视频格式和播放应用程序中Alpha通道的支持，并设计了一种融合算法来确保视觉隐形性和兼容性。我们在五个最先进的对象检测器、视觉语言模型和多模式大型语言模型（Gemini-2.0-Flash）上评估了{\Alpha}-Cloak，展示了在所有场景下100%的攻击成功率。我们的研究结果揭示了基于视频的感知系统中以前未探索的漏洞，凸显了对对抗环境中阿尔法通道的防御的迫切需要。



## **44. FPT-Noise: Dynamic Scene-Aware Counterattack for Test-Time Adversarial Defense in Vision-Language Models**

FPT噪声：视觉语言模型中测试时对抗防御的动态场景感知反击 cs.CR

11pages,4figures

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.20856v1) [paper-pdf](http://arxiv.org/pdf/2510.20856v1)

**Authors**: Jia Deng, Jin Li, Zhenhua Zhao, Shaowei Wang

**Abstract**: Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot generalizability across diverse downstream tasks. However, recent studies have revealed that VLMs, including CLIP, are highly vulnerable to adversarial attacks, particularly on their visual modality. Traditional methods for improving adversarial robustness, such as adversarial training, involve extensive retraining and can be computationally expensive. In this paper, we propose a new Test-Time defense: Feature Perception Threshold Counterattack Noise (FPT-Noise), which enhances the adversarial robustness of CLIP without costly fine-tuning. Our core contributions are threefold: First, we introduce a Dynamic Feature Modulator that dynamically generate an image-specific and attack-adaptive noise intensity parameter. Second, We reanalyzed the image features of CLIP. When images are exposed to different levels of noise, clean images and adversarial images exhibit distinct rates of feature change. We established a feature perception threshold to distinguish clean images from attacked ones. Finally, we integrate a Scene-Aware Regulation guided by a stability threshold and leverage Test-Time Transformation Ensembling (TTE) to further mitigate the impact of residual noise and enhance robustness.Extensive experimentation has demonstrated that FPT-Noise significantly outperforms existing Test-Time defense methods, boosting average robust accuracy from 0.07% to 56.86% under AutoAttack while maintaining high performance on clean images (-1.1%). The code will be made public following the publication of the study. The code will be made public following the publication of the study.

摘要: CLIP等视觉语言模型（VLM）已在各种下游任务中表现出出色的零射击通用性。然而，最近的研究表明，包括CLIP在内的VLM非常容易受到对抗攻击，特别是在其视觉形态上。提高对抗鲁棒性的传统方法（例如对抗训练）涉及大量的再培训，并且计算成本可能很高。在本文中，我们提出了一种新的测试时防御：特征感知阈值反击噪音（FPT-Noise），它增强了CLIP的对抗鲁棒性，而无需进行昂贵的微调。我们的核心贡献有三方面：首先，我们引入了动态特征调制器，它动态生成特定于图像和攻击自适应的噪音强度参数。其次，我们重新分析了CLIP的图像特征。当图像暴露于不同水平的噪音时，干净图像和对抗图像会表现出不同的特征变化率。我们建立了一个特征感知阈值来区分干净的图像和受攻击的图像。最后，我们集成了以稳定性阈值为指导的场景感知监管，并利用测试时转换集成（TTE）来进一步减轻残余噪音的影响并增强稳健性。大量实验表明，FPT-Noise显着优于现有的测试时防御方法，在AutoAttack下，将平均稳健准确性从0.07%提高到56.86%，同时在干净图像上保持高性能（-1.1%）。该准则将在研究发布后公开。该准则将在研究发布后公开。



## **45. A New Type of Adversarial Examples**

新型对抗性例子 cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19347v1) [paper-pdf](http://arxiv.org/pdf/2510.19347v1)

**Authors**: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang

**Abstract**: Most machine learning models are vulnerable to adversarial examples, which poses security concerns on these models. Adversarial examples are crafted by applying subtle but intentionally worst-case modifications to examples from the dataset, leading the model to output a different answer from the original example. In this paper, adversarial examples are formed in an exactly opposite manner, which are significantly different from the original examples but result in the same answer. We propose a novel set of algorithms to produce such adversarial examples, including the negative iterative fast gradient sign method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM), along with their momentum variants: the negative momentum iterative fast gradient sign method (NMI-FGSM) and the negative momentum iterative fast gradient method (NMI-FGM). Adversarial examples constructed by these methods could be used to perform an attack on machine learning systems in certain occasions. Moreover, our results show that the adversarial examples are not merely distributed in the neighbourhood of the examples from the dataset; instead, they are distributed extensively in the sample space.

摘要: 大多数机器学习模型都容易受到对抗性示例的影响，这对这些模型带来了安全问题。对抗性示例是通过对数据集中的示例应用微妙但故意最坏情况的修改来精心设计的，导致模型输出与原始示例不同的答案。在本文中，对抗性例子以完全相反的方式形成，与原始例子显着不同，但得到相同的答案。我们提出了一组新颖的算法来产生此类对抗性示例，包括负迭代快速梯度符号法（NI-FGSM）和负迭代快速梯度法（NI-FGM），以及它们的动量变体：负动量迭代快速梯度符号法（NMI-FGSM）和负动量迭代快速梯度法（NMI-FGM）。通过这些方法构建的对抗性示例可以在某些情况下用于对机器学习系统进行攻击。此外，我们的结果表明，对抗性示例不仅分布在数据集示例的附近;相反，它们广泛分布在样本空间中。



## **46. Collaborative penetration testing suite for emerging generative AI algorithms**

针对新兴生成式人工智能算法的协作渗透测试套件 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19303v1) [paper-pdf](http://arxiv.org/pdf/2510.19303v1)

**Authors**: Petar Radanliev

**Abstract**: Problem Space: AI Vulnerabilities and Quantum Threats Generative AI vulnerabilities: model inversion, data poisoning, adversarial inputs. Quantum threats Shor Algorithm breaking RSA ECC encryption. Challenge Secure generative AI models against classical and quantum cyberattacks. Proposed Solution Collaborative Penetration Testing Suite Five Integrated Components: DAST SAST OWASP ZAP, Burp Suite, SonarQube, Fortify. IAST Contrast Assess integrated with CI CD pipeline. Blockchain Logging Hyperledger Fabric for tamper-proof logs. Quantum Cryptography Lattice based RLWE protocols. AI Red Team Simulations Adversarial ML & Quantum-assisted attacks. Integration Layer: Unified workflow for AI, cybersecurity, and quantum experts. Key Results 300+ vulnerabilities identified across test environments. 70% reduction in high-severity issues within 2 weeks. 90% resolution efficiency for blockchain-logged vulnerabilities. Quantum-resistant cryptography maintained 100% integrity in tests. Outcome: Quantum AI Security Protocol integrating Blockchain Quantum Cryptography AI Red Teaming.

摘要: 问题空间：人工智能漏洞和量子威胁生成人工智能漏洞：模型倒置、数据中毒、对抗性输入。量子威胁Shor算法破解RSA椭圆曲线加密。挑战确保生成性人工智能模型免受经典和量子网络攻击。提议的解决方案协作渗透测试套件五个集成组件：DAST SAST OWASP ZAP、Burp Suite、SonarQube、Fortify。IAST对比评估与CI CD管道集成。区块链日志Hyperledger结构，用于防篡改日志。基于量子密码格子的RLWE协议。AI Red Team模拟对抗ML和量子辅助攻击。集成层：针对人工智能、网络安全和量子专家的统一工作流程。关键结果测试环境中发现了300多个漏洞。2周内高严重性问题减少70%。区块链记录漏洞的解决效率为90%。抗量子加密技术在测试中保持了100%的完整性。结果：集成区块链量子加密技术的量子AI安全协议AI Red团队。



## **47. Adversarial Attacks on LiDAR-Based Tracking Across Road Users: Robustness Evaluation and Target-Aware Black-Box Method**

对基于LiDART的跨路跟踪用户的对抗攻击：稳健性评估和目标感知黑匣子方法 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2410.20893v3) [paper-pdf](http://arxiv.org/pdf/2410.20893v3)

**Authors**: Shengjing Tian, Xiantong Zhao, Yuhao Bian, Yinan Han, Bin Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDART点云跟踪模型在对抗性攻击下的稳健性，这是一个经常被忽视的关键方面，而有利于性能增强。尽管这些模型采用了Transformer或Bird ' s Eye View（BEV）等先进架构，但在面对对抗性攻击、域转移或数据损坏等挑战时，往往会忽视稳健性。相反，我们关注的是跟踪模型在对抗攻击威胁下的稳健性。我们首先建立一个统一的框架，用于在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑匣子攻击策略。对于白盒攻击，我们定制特定的损失函数以适应各种跟踪范式，并将FGSM、C\& W和PVD等现有方法扩展到点云域。在解决黑匣子攻击场景时，我们引入了一种新颖的基于传输的方法，即目标感知扰动生成（TAPG）算法，其双重目标是实现高攻击性能和保持低感知性。该方法采用启发式策略来强制稀疏攻击约束，并利用随机子载体分解来增强可移植性。我们的实验结果揭示了高级跟踪方法在遭受黑匣子和白盒攻击时存在重大漏洞，强调了将对抗攻击的鲁棒性纳入LiDART点云跟踪模型的设计中的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和干扰的隐藏之间取得了最佳平衡。



## **48. Defending Against Prompt Injection with DataFilter**

使用数据过滤器防御提示注入 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.

摘要: 当大型语言模型（LLM）代理越来越多地被部署来自动化任务并与不受信任的外部数据交互时，即时注入成为一个重大的安全威胁。通过将恶意指令注入LLM访问的数据中，攻击者可以任意覆盖原始用户任务，并将代理重定向到无意的、可能有害的操作。现有的防御要么需要访问模型权重（微调），导致大量效用损失（基于检测），要么要求进行非平凡的系统重新设计（系统级）。出于此动机，我们提出了数据过滤器，这是一种测试时模型不可知的防御，可以在数据到达后台LLM之前从数据中删除恶意指令。数据过滤器通过模拟注入的监督微调进行训练，并利用用户的指令和数据来选择性地剥离对抗内容，同时保留良性信息。在多个基准测试中，数据过滤器一致地将即时注入攻击成功率降低到接近零，同时保持LLM的实用性。数据过滤器提供强大的安全性、高实用性和即插即用部署，使其成为保护黑匣子商业LLM免受即时注入的强大实用防御。我们的数据过滤器模型已在https://huggingface.co/JoyYizhu/DataFilter上发布，可立即使用，其代码可在https://github.com/yizhu-joy/DataFilter上重现我们的结果。



## **49. FeatureFool: Zero-Query Fooling of Video Models via Feature Map**

DeliverureFool：通过特征图对视频模型进行零查询愚弄 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.18362v2) [paper-pdf](http://arxiv.org/pdf/2510.18362v2)

**Authors**: Duoxun Tang, Xi Xiao, Guangwu Hu, Kangkang Sun, Xiao Yang, Dongyang Chen, Qing Li, Yongjie Yin, Jiyao Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.

摘要: 深度神经网络（DNN）的脆弱性已初步得到验证。现有的黑匣子对抗攻击通常需要与模型进行多轮交互并消耗大量查询，这在现实世界中是不切实际的，并且很难扩展到最近出现的Video-LLM。此外，视频领域中没有任何攻击直接利用特征地图来移动干净视频特征空间。因此，我们提出了DeliverureFool，这是一种隐形的、视频域的、零查询黑匣子攻击，利用从DNN提取的信息来改变干净视频的特征空间。与依赖于迭代交互的基于查询的方法不同，InspectureFool通过直接利用DNN提取的信息来执行零查询攻击。这种高效的方法在视频领域是前所未有的。实验表明，在没有任何查询的情况下，DeliverureFool针对传统视频分类器的攻击成功率超过70%。受益于特征地图的可移植性，它还可以制作有害内容并绕过Video-LLM识别。此外，DeliverureFool生成的对抗视频在SSIM、PSNR和时间不一致性方面表现出高质量，使攻击几乎不可察觉。本文可能包含暴力或露骨内容。



## **50. The Black Tuesday Attack: how to crash the stock market with adversarial examples to financial forecasting models**

黑色星期二攻击：如何通过金融预测模型的对抗例子来崩溃股市 cs.CR

15 pages, 2 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18990v1) [paper-pdf](http://arxiv.org/pdf/2510.18990v1)

**Authors**: Thomas Hofweber, Jefrey Bergl, Ian Reyes, Amir Sadovnik

**Abstract**: We investigate and defend the possibility of causing a stock market crash via small manipulations of individual stock values that together realize an adversarial example to financial forecasting models, causing these models to make the self-fulfilling prediction of a crash. Such a crash triggered by an adversarial example would likely be hard to detect, since the model's predictions would be accurate and the interventions that would cause it are minor. This possibility is a major risk to financial stability and an opportunity for hostile actors to cause great economic damage to an adversary. This threat also exists against individual stocks and the corresponding valuation of individual companies. We outline how such an attack might proceed, what its theoretical basis is, how it can be directed towards a whole economy or an individual company, and how one might defend against it. We conclude that this threat is vastly underappreciated and requires urgent research on how to defend against it.

摘要: 我们调查并捍卫通过对个别股票价值的小额操纵而导致股市崩盘的可能性，这些操纵共同实现了金融预测模型的对抗性例子，导致这些模型对崩盘做出自我实现的预测。由敌对例子引发的这种崩溃可能很难检测到，因为模型的预测是准确的，而且导致崩溃的干预措施也很小。这种可能性是金融稳定的重大风险，也是敌对行为者对对手造成巨大经济损害的机会。这种威胁也存在于个别股票和个别公司的相应估值。我们概述了这种攻击可能如何进行、其理论基础是什么、如何针对整个经济或单个公司以及如何防御它。我们的结论是，这种威胁被严重低估，需要紧急研究如何防御它。



