# Latest Adversarial Attack Papers
**update at 2025-11-19 09:21:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning**

针对两个对手进行调优：使用超参数调优增强针对传输和基于查询的攻击的鲁棒性 cs.LG

To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13654v1) [paper-pdf](https://arxiv.org/pdf/2511.13654v1)

**Authors**: Pascal Zimmer, Ghassan Karame

**Abstract**: In this paper, we present the first detailed analysis of how optimization hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the optimization hyperparameter design space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.

摘要: 在本文中，我们首次详细分析了优化超参数（例如学习率、权重衰减、动量和批量大小）如何影响针对基于传输和基于查询的攻击的鲁棒性。在理论和实验的支持下，我们的研究涵盖了各种实际部署环境，包括集中式培训、集成学习和分布式培训。我们发现了一个引人注目的二分法：对于基于传输的攻击，降低学习率可以显着增强鲁棒性，提高高达64美元。相比之下，对于基于查询的攻击，在各种设置和数据分布中持续提高学习率可使稳健性提高高达28美元。利用这些发现，我们首次探索优化超参数设计空间，以共同增强针对基于传输和基于查询的攻击的鲁棒性。我们的结果表明，分布式模型从超参数调整中受益最多，通过比其他训练设置更有效地同时减轻两种攻击类型，实现了显着的权衡。



## **2. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

ForgeDAN：越狱对齐大型语言模型的进化框架 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.

摘要: 大型语言模型（LLM）的迅速采用既带来了变革性的应用程序，也带来了新的安全风险，包括绕过对齐保障措施以引发有害输出的越狱攻击。现有的自动越狱生成方法（例如AutoDAN）存在突变多样性有限、适应度评估浅和基于关键字的脆弱检测的问题。为了解决这些限制，我们提出了ForgeDAN，这是一种新颖的进化框架，用于针对对齐的LLM生成语义一致且高效的对抗性提示。首先，ForgeDAN在\textit{字符、单词和会话级别}操作中引入多策略文本扰动，以增强攻击多样性;然后我们基于文本相似性模型采用可解释的语义适应度评估来引导进化过程走向语义相关和有害的输出;最后，ForgeDAN集成了二维越狱判断，利用基于LLM的分类器来联合评估模型合规性和输出危害性，从而减少假阳性并提高检测有效性。我们的评估表明，ForgeDAN在保持自然性和隐形性的同时实现了很高的越狱成功率，优于现有的SOTA解决方案。



## **3. Robust Defense Strategies for Multimodal Contrastive Learning: Efficient Fine-tuning Against Backdoor Attacks**

多模式对比学习的稳健防御策略：针对后门攻击的有效微调 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13545v1) [paper-pdf](https://arxiv.org/pdf/2511.13545v1)

**Authors**: Md. Iqbal Hossain, Afia Sajeeda, Neeresh Kumar Perla, Ming Shao

**Abstract**: The advent of multimodal deep learning models, such as CLIP, has unlocked new frontiers in a wide range of applications, from image-text understanding to classification tasks. However, these models are not safe for adversarial attacks, particularly backdoor attacks, which can subtly manipulate model behavior. Moreover, existing defense methods typically involve training from scratch or fine-tuning using a large dataset without pinpointing the specific labels that are affected. In this study, we introduce an innovative strategy to enhance the robustness of multimodal contrastive learning models against such attacks. In particular, given a poisoned CLIP model, our approach can identify the backdoor trigger and pinpoint the victim samples and labels in an efficient manner. To that end, an image segmentation ``oracle'' is introduced as the supervisor for the output of the poisoned CLIP. We develop two algorithms to rectify the poisoned model: (1) differentiating between CLIP and Oracle's knowledge to identify potential triggers; (2) pinpointing affected labels and victim samples, and curating a compact fine-tuning dataset. With this knowledge, we are allowed to rectify the poisoned CLIP model to negate backdoor effects. Extensive experiments on visual recognition benchmarks demonstrate our strategy is effective in CLIP-based backdoor defense.

摘要: CLIP等多模式深度学习模型的出现打开了从图像文本理解到分类任务等广泛应用的新领域。然而，这些模型对于对抗攻击（尤其是后门攻击）并不安全，因为后门攻击可以微妙地操纵模型行为。此外，现有的防御方法通常涉及从头开始训练或使用大型数据集进行微调，而不确定受影响的特定标签。在这项研究中，我们引入了一种创新的策略来增强多模式对比学习模型针对此类攻击的鲁棒性。特别是，给定有毒的CLIP模型，我们的方法可以识别后门触发器并以有效的方式确定受害者样本和标签。为此，引入了图像分割“oracle”作为中毒CLIP输出的监督器。我们开发了两种算法来纠正中毒模型：（1）区分CLIP和Oracle的知识以识别潜在的触发因素;（2）确定受影响的标签和受害者样本，并策划紧凑的微调数据集。有了这些知识，我们就可以纠正有毒的CLIP模型以抵消后门效应。视觉识别基准的大量实验表明我们的策略在基于CLIP的后门防御中是有效的。



## **4. Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew**

准确性还不够：颜色倾斜毒害联邦学习中的解释性 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13535v1) [paper-pdf](https://arxiv.org/pdf/2511.13535v1)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska

**Abstract**: As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.

摘要: 随着机器学习模型越来越多地部署在安全关键领域，视觉解释技术已成为支持透明度的重要工具。在这项工作中，我们揭示了一类新型攻击，这些攻击在不影响准确性的情况下损害了模型的可解释性。具体来说，我们表明，在联邦学习环境中，对抗客户端应用的小颜色扰动可以将模型的显着性地图从语义有意义的区域移开，同时保持预测不变。提出的显着性感知攻击框架称为色彩扰动模块，通过以破坏解释保真度的方式改变前景和背景之间的颜色对比度，系统性地制作对抗性示例。这些扰动在训练轮中累积，以一种隐秘且持续的方式毒害了全球模型的内部特征属性。我们的发现挑战了模型审计中的一个常见假设，即正确的预测意味着忠实的解释，并证明可解释性本身可能是一种攻击面。我们在多个数据集中评估了这个漏洞，并表明标准训练管道不足以检测或减轻解释退化，特别是在联邦学习环境中，微妙的颜色扰动更难辨别。我们的攻击将Grad-CAM解释中的峰值激活重叠减少了高达35%，同时在所有评估的数据集上将分类准确率保持在96%以上。



## **5. Cyber-Resilient Fault Diagnosis Methodology in Inverter-Based Resource-Dominated Microgrids with Single-Point Measurement**

基于逆变器的资源主导微电网单点测量的网络弹性故障诊断方法 eess.SY

5 pages, 5 figures

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13162v1) [paper-pdf](https://arxiv.org/pdf/2511.13162v1)

**Authors**: Yifan Wang, Yiyao Yu, Yang Xia, Yan Xu

**Abstract**: Cyber-attacks jeopardize the safe operation of inverter-based resource-dominated microgrids (IBR-dominated microgrids). At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modeling assumptions that are untenable under single-point measurement constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves timely fault localization and cyber-resilient fault diagnosis using only one VPQ (voltage, active power, reactive power) measurement point. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Grünwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritize the most challenging samples. Experiments on a four-inverter IBR-dominated microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6% (bias), 94.0% (noise), 92.8% (data replacement), and 95.7% (replay), while sustaining 96.7% under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of IBR-dominated microgrids.

摘要: 网络攻击危及基于逆变器的资源主导微电网（IBR主导微电网）的安全运行。与此同时，现有的诊断方法要么依赖于昂贵的多点仪器，要么依赖于在单点测量约束下站不住脚的严格建模假设。本文提出了一种分数阶记忆增强型攻击诊断方案（FO-MADS），仅使用一个VPQ（电压、有效功率、无功功率）测量点即可实现及时的故障定位和网络弹性故障诊断。FO-MADS首先通过联合应用Caputo和Grünwald-Letnikov导数来构建双分数阶特征库，从而放大VPQ信号中的微扰动和缓慢漂移。一个两阶段的分层分类器，然后查明受影响的逆变器和隔离故障IGBT开关，有效地缓解类不平衡。通过渐进记忆回放对抗训练（PMR-AT）进一步加强鲁棒性，其攻击感知损失通过在线硬示例挖掘（OHEEM）动态重新加权，以优先考虑最具挑战性的样本。在四种攻击场景下由1个正常类别和24个故障类别组成的四个逆变器IBR主导的微电网测试台上进行的实验表明，诊断准确率为96.6%（偏差）、94.0%（噪音）、92.8%（数据替换）和95.7%（回放），而在无攻击条件下保持96.7%。这些结果使FO-MADS成为一种具有成本效益且易于部署的解决方案，可以显着增强IBR主导的微电网的网络物理弹性。



## **6. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

VLN鲁棒性的减弱：基于室内照明的对抗攻击的黑匣子框架 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.

摘要: 视觉与语言导航（VLN）代理已经取得了显着的进步，但其稳健性仍然研究不足。现有的对抗性评估通常依赖于扰动，这些扰动表现为日常室内环境中很少遇到的异常纹理。这种人为条件下的错误的实际意义有限，因为现实世界的代理人不太可能遇到这种人为模式。在这项工作中，我们重点关注室内照明，这是一种固有但在很大程度上被忽视的场景属性，它强烈影响导航。我们提出了基于室内照明的对抗攻击（ILA），这是一种黑匣子框架，可以操纵全球照明来扰乱VLN代理。受典型家庭照明使用的启发，我们设计了两种攻击模式：静态室内照明攻击（SILA），其中照明强度在整个剧集中保持恒定，以及动态室内照明攻击（DILA），其中在关键时刻打开或关闭灯光以引发突然的照明变化。我们在三个导航任务中评估了两个最先进的VLN模型的ILA。结果表明，ILA显着增加了故障率，同时降低了轨迹效率，揭示了VLN代理对现实室内照明变化的脆弱性。



## **7. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

VEIL：通过隐性语言的视觉开发破解文本到视频模型 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13127v1) [paper-pdf](https://arxiv.org/pdf/2511.13127v1)

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models.

摘要: 越狱攻击可以绕过模型安全护栏并暴露关键盲点。先前对文本转视频（T2 V）模型的攻击通常会向明显不安全的提示添加对抗性扰动，而这些提示通常很容易检测和防御。相比之下，我们表明，包含丰富、隐性线索的看似友善的提示可以诱导T2 V模型生成语义不安全的视频，这些视频既违反了政策，又保留了原始（被阻止的）意图。为了实现这一点，我们提出了VEIL，这是一个越狱框架，通过模块化提示设计利用T2 V模型的跨模式关联模式。具体来说，我们的提示结合了三个组件：中性场景锚点，它提供从被阻止的意图中提取的表面级场景描述，以保持可信性;潜在听觉触发器，听起来无害的音频事件的文本描述（例如，吱吱作响、低沉的噪音），利用习得的视听同现先验来将模型偏向特定不安全的视觉概念;以及风格调节器、电影指令（例如，相机取景、大气）放大和稳定潜在触发的效果。我们将攻击生成形式化为对上述模块提示空间的约束优化，并通过平衡隐形性和有效性的引导搜索过程来解决它。对7个T2 V模型进行的广泛实验证明了我们攻击的有效性，使商业模型中的平均攻击成功率提高了23%。



## **8. Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks**

角梯度符号法：揭示双曲网络中的漏洞 cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.12985v1) [paper-pdf](https://arxiv.org/pdf/2511.12985v1)

**Authors**: Minsoo Jo, Dongyoon Yang, Taesup Kim

**Abstract**: Adversarial examples in neural networks have been extensively studied in Euclidean geometry, but recent advances in \textit{hyperbolic networks} call for a reevaluation of attack strategies in non-Euclidean geometries. Existing methods such as FGSM and PGD apply perturbations without regard to the underlying hyperbolic structure, potentially leading to inefficient or geometrically inconsistent attacks. In this work, we propose a novel adversarial attack that explicitly leverages the geometric properties of hyperbolic space. Specifically, we compute the gradient of the loss function in the tangent space of hyperbolic space, decompose it into a radial (depth) component and an angular (semantic) component, and apply perturbation derived solely from the angular direction. Our method generates adversarial examples by focusing perturbations in semantically sensitive directions encoded in angular movement within the hyperbolic geometry. Empirical results on image classification, cross-modal retrieval tasks and network architectures demonstrate that our attack achieves higher fooling rates than conventional adversarial attacks, while producing high-impact perturbations with deeper insights into vulnerabilities of hyperbolic embeddings. This work highlights the importance of geometry-aware adversarial strategies in curved representation spaces and provides a principled framework for attacking hierarchical embeddings.

摘要: 欧几里德几何中对神经网络中的对抗示例进行了广泛研究，但双曲网络的最新进展需要重新评估非欧几里德几何中的攻击策略。FGSM和PVD等现有方法应用扰动，而不考虑潜在的双曲结构，这可能会导致低效或几何不一致的攻击。在这项工作中，我们提出了一种新颖的对抗攻击，它明确利用了双曲空间的几何性质。具体来说，我们计算双曲空间的切空间中损失函数的梯度，将其分解为半径（深度）分量和角度（语义）分量，并应用仅从角度方向派生的扰动。我们的方法通过将扰动集中在双曲几何内的角运动中编码的语义敏感方向上来生成对抗性示例。图像分类、跨模式检索任务和网络架构的经验结果表明，我们的攻击比传统的对抗性攻击获得了更高的愚弄率，同时通过更深入地了解双曲嵌入的脆弱性，产生了高影响力的扰动。这项工作强调了弯曲表示空间中几何感知对抗策略的重要性，并为攻击分层嵌入提供了一个原则性框架。



## **9. T2I-Based Physical-World Appearance Attack against Traffic Sign Recognition Systems in Autonomous Driving**

针对自动驾驶中交通标志识别系统的基于T2 I的物理世界外观攻击 cs.CV

16 pages, 12 figures

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.12956v1) [paper-pdf](https://arxiv.org/pdf/2511.12956v1)

**Authors**: Chen Ma, Ningfei Wang, Junhao Zheng, Qing Guo, Qian Wang, Qi Alfred Chen, Chao Shen

**Abstract**: Traffic Sign Recognition (TSR) systems play a critical role in Autonomous Driving (AD) systems, enabling real-time detection of road signs, such as STOP and speed limit signs. While these systems are increasingly integrated into commercial vehicles, recent research has exposed their vulnerability to physical-world adversarial appearance attacks. In such attacks, carefully crafted visual patterns are misinterpreted by TSR models as legitimate traffic signs, while remaining inconspicuous or benign to human observers. However, existing adversarial appearance attacks suffer from notable limitations. Pixel-level perturbation-based methods often lack stealthiness and tend to overfit to specific surrogate models, resulting in poor transferability to real-world TSR systems. On the other hand, text-to-image (T2I) diffusion model-based approaches demonstrate limited effectiveness and poor generalization to out-of-distribution sign types.   In this paper, we present DiffSign, a novel T2I-based appearance attack framework designed to generate physically robust, highly effective, transferable, practical, and stealthy appearance attacks against TSR systems. To overcome the limitations of prior approaches, we propose a carefully designed attack pipeline that integrates CLIP-based loss and masked prompts to improve attack focus and controllability. We also propose two novel style customization methods to guide visual appearance and improve out-of-domain traffic sign attack generalization and attack stealthiness. We conduct extensive evaluations of DiffSign under varied real-world conditions, including different distances, angles, light conditions, and sign categories. Our method achieves an average physical-world attack success rate of 83.3%, leveraging DiffSign's high effectiveness in attack transferability.

摘要: 交通标志识别（TSB）系统在自动驾驶（AD）系统中发挥着关键作用，能够实时检测道路标志，例如停止和限速标志。虽然这些系统越来越多地集成到商用车辆中，但最近的研究暴露了它们容易受到物理世界对抗外观攻击的脆弱性。在此类攻击中，精心制作的视觉模式被TSB模型误解为合法的交通标志，而对人类观察者来说仍然不引人注目或无害。然而，现有的对抗性外观攻击存在显着的局限性。基于像素级扰动的方法通常缺乏隐蔽性，并且往往过度适合特定的代理模型，导致到现实世界的TSB系统的可移植性较差。另一方面，基于文本到图像（T2 I）扩散模型的方法表现出有效性有限且对非分布标志类型的概括性较差。   在本文中，我们提出了一种新型的基于T2 I的外观攻击框架，旨在针对TSB系统生成物理上稳健、高效、可转移、实用且隐蔽的外观攻击。为了克服现有方法的局限性，我们提出了一种精心设计的攻击管道，该管道集成了基于CLIP的丢失和掩蔽提示，以提高攻击焦点和可控性。我们还提出了两种新颖的风格定制方法来引导视觉外观并提高域外交通标志攻击的概括性和攻击隐蔽性。我们在各种现实条件下（包括不同的距离、角度、光线条件和标牌类别）对迪夫Sign进行广泛评估。我们的方法利用了迪夫Sign在攻击可转移性方面的高效率，实现了平均物理世界攻击成功率83.3%。



## **10. Efficient Adversarial Malware Defense via Trust-Based Raw Override and Confidence-Adaptive Bit-Depth Reduction**

通过基于信任的原始数据库和保密自适应位深缩减实现高效的对抗性恶意软件防御 cs.CR

Accepted at IEEE International Conference on Big Data 2025. 10 pages, 2 figures, 8 tables

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12827v1) [paper-pdf](https://arxiv.org/pdf/2511.12827v1)

**Authors**: Ayush Chaudhary, Sisir Doppalpudi

**Abstract**: The deployment of robust malware detection systems in big data environments requires careful consideration of both security effectiveness and computational efficiency. While recent advances in adversarial defenses have demonstrated strong robustness improvements, they often introduce computational overhead ranging from 4x to 22x, which presents significant challenges for production systems processing millions of samples daily. In this work, we propose a novel framework that combines Trust-Raw Override (TRO) with Confidence-Adaptive Bit-Depth Reduction (CABDR) to explicitly optimize the trade-off between adversarial robustness and computational efficiency. Our approach leverages adaptive confidence-based mechanisms to selectively apply defensive measures, achieving 1.76x computational overhead - a 2.3x improvement over state-of-the-art smoothing defenses. Through comprehensive evaluation on the EMBER v2 dataset comprising 800K samples, we demonstrate that our framework maintains 91 percent clean accuracy while reducing attack success rates to 31-37 percent across multiple attack types, with particularly strong performance against optimization-based attacks such as C and W (48.8 percent reduction). The framework achieves throughput of up to 1.26 million samples per second (measured on pre-extracted EMBER features with no runtime feature extraction), validated across 72 production configurations with statistical significance (5 independent runs, 95 percent confidence intervals, p less than 0.01). Our results suggest that practical adversarial robustness in production environments requires explicit optimization of the efficiency-robustness trade-off, providing a viable path for organizations to deploy robust defenses without prohibitive infrastructure costs.

摘要: 在大数据环境中部署强大的恶意软件检测系统需要仔细考虑安全有效性和计算效率。虽然对抗性防御的最新进展显示出强大的鲁棒性改进，但它们通常会带来4倍到22倍的计算负担，这给每天处理数百万个样本的生产系统带来了重大挑战。在这项工作中，我们提出了一种新颖的框架，该框架将信任原始搜索器（TRO）与信任自适应比特深度缩减（CABER）相结合，以显式优化对抗鲁棒性和计算效率之间的权衡。我们的方法利用基于自适应信任的机制来选择性地应用防御措施，实现了1.76倍的计算负担--比最先进的平滑防御提高了2.3倍。通过对包含80万个样本的EBER v2数据集的全面评估，我们证明我们的框架保持了91%的干净准确性，同时将多种攻击类型的攻击成功率降低到31%-37%，针对基于优化的攻击（例如C和W）的性能尤其强劲（减少48.8%）。该框架实现了每秒高达126万个样本的吞吐量（根据预提取的EBER特征进行测量，无需运行时特征提取），并在72种具有统计学意义的生产配置中进行了验证（5次独立运行，95%置信区间，p小于0.01）。我们的结果表明，生产环境中实际的对抗稳健性需要对效率-稳健性权衡进行显式优化，为组织提供一种可行的途径，在不承担高昂的基础设施成本的情况下部署稳健的防御。



## **11. LLM Reinforcement in Context**

LLM在上下文中的强化 cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.

摘要: 当前的大型语言模型对齐研究主要集中在通过对示例和提示进行训练来提高模型对对抗性攻击和不当行为的稳健性。研究表明，LLM越狱概率随着用户输入或对话长度的大小而增加。缺乏对加强对齐的方法进行适当的研究，而对齐也随用户输入长度而变化。我们建议中断作为这个问题的一种可能的解决方案。中断是针对某个任意x，大约每x个记号添加到用户输入中的控制句。我们建议将其推广到思想链过程中，以防止阴谋。



## **12. On Robustness of Linear Classifiers to Targeted Data Poisoning**

线性分类器对目标数据中毒的鲁棒性 cs.LG

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12722v1) [paper-pdf](https://arxiv.org/pdf/2511.12722v1)

**Authors**: Nakshatra Gupta, Sumanth Prabhu, Supratik Chakraborty, R Venkatesh

**Abstract**: Data poisoning is a training-time attack that undermines the trustworthiness of learned models. In a targeted data poisoning attack, an adversary manipulates the training dataset to alter the classification of a targeted test point. Given the typically large size of training dataset, manual detection of poisoning is difficult. An alternative is to automatically measure a dataset's robustness against such an attack, which is the focus of this paper. We consider a threat model wherein an adversary can only perturb the labels of the training dataset, with knowledge limited to the hypothesis space of the victim's model. In this setting, we prove that finding the robustness is an NP-Complete problem, even when hypotheses are linear classifiers. To overcome this, we present a technique that finds lower and upper bounds of robustness. Our implementation of the technique computes these bounds efficiently in practice for many publicly available datasets. We experimentally demonstrate the effectiveness of our approach. Specifically, a poisoning exceeding the identified robustness bounds significantly impacts test point classification. We are also able to compute these bounds in many more cases where state-of-the-art techniques fail.

摘要: 数据中毒是一种训练时攻击，会破坏学习模型的可信度。在有针对性的数据中毒攻击中，对手操纵训练数据集以改变目标测试点的分类。鉴于训练数据集通常很大，手动检测中毒很困难。一种替代方案是自动测量数据集对此类攻击的稳健性，这是本文的重点。我们考虑一个威胁模型，其中对手只能扰乱训练数据集的标签，知识仅限于受害者模型的假设空间。在这种情况下，我们证明即使假设是线性分类器，找到稳健性也是一个NP完全问题。为了克服这个问题，我们提出了一种找到稳健性下限和上限的技术。我们对该技术的实现在实践中有效地计算了许多公开可用的数据集的这些界限。我们通过实验证明了我们方法的有效性。具体来说，超过已确定的稳健性界限的中毒会显着影响测试点分类。我们还能够在更多最先进技术失败的情况下计算这些界限。



## **13. Scalable Hierarchical AI-Blockchain Framework for Real-Time Anomaly Detection in Large-Scale Autonomous Vehicle Networks**

用于大规模自主车辆网络中实时异常检测的可扩展分层AI区块链框架 cs.CR

Submitted to the Journal

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12648v1) [paper-pdf](https://arxiv.org/pdf/2511.12648v1)

**Authors**: Rathin Chandra Shit, Sharmila Subudhi

**Abstract**: The security of autonomous vehicle networks is facing major challenges, owing to the complexity of sensor integration, real-time performance demands, and distributed communication protocols that expose vast attack surfaces around both individual and network-wide safety. Existing security schemes are unable to provide sub-10 ms (milliseconds) anomaly detection and distributed coordination of large-scale networks of vehicles within an acceptable safety/privacy framework. This paper introduces a three-tier hybrid security architecture HAVEN (Hierarchical Autonomous Vehicle Enhanced Network), which decouples real-time local threat detection and distributed coordination operations. It incorporates a light ensemble anomaly detection model on the edge (first layer), Byzantine-fault-tolerant federated learning to aggregate threat intelligence at a regional scale (middle layer), and selected blockchain mechanisms (top layer) to ensure critical security coordination. Extensive experimentation is done on a real-world autonomous driving dataset. Large-scale simulations with the number of vehicles ranging between 100 and 1000 and different attack types, such as sensor spoofing, jamming, and adversarial model poisoning, are conducted to test the scalability and resiliency of HAVEN. Experimental findings show sub-10 ms detection latency with an accuracy of 94% and F1-score of 92% across multimodal sensor data, Byzantine fault tolerance validated with 20\% compromised nodes, and a reduced blockchain storage overhead, guaranteeing sufficient differential privacy. The proposed framework overcomes the important trade-off between real-time safety obligation and distributed security coordination with novel three-tiered processing. The scalable architecture of HAVEN is shown to provide great improvement in detection accuracy as well as network resilience over other methods.

摘要: 由于传感器集成的复杂性、实时性能要求以及分布式通信协议，自动驾驶汽车网络的安全性面临着重大挑战，这些协议暴露了围绕个人和网络安全的巨大攻击面。现有的安全方案不能在可接受的安全/隐私框架内提供10 ms（毫秒）以下的异常检测和大规模车辆网络的分布式协调。本文介绍了一种三层混合安全体系结构HAVEN（Hierarchical Autonomous Vehicle Enhanced Network），它将实时局部威胁检测和分布式协调操作相结合。它在边缘（第一层）上集成了一个轻型集成异常检测模型，拜占庭容错联邦学习在区域范围内聚合威胁情报（中间层），并选择区块链机制（顶层）以确保关键的安全协调。在现实世界的自动驾驶数据集上进行了广泛的实验。对车辆数量在100至1000之间以及不同攻击类型（例如传感器欺骗、干扰和对抗模型中毒）进行大规模模拟，以测试HAVEN的可扩展性和弹性。实验结果表明，多模式传感器数据的检测延迟低于10 ms，准确率为94%，F1评分为92%，Byzantine的故障容限通过20%的受影响节点进行验证，并且降低了区块链存储负担，保证了足够的差异隐私。提出的框架通过新颖的三层处理克服了实时安全义务和分布式安全协调之间的重要权衡。事实证明，HAVEN的可扩展架构比其他方法在检测准确性和网络弹性方面提供了很大的改进。



## **14. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

超越像素：用于地理隐私保护的语义感知印刷攻击 cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.

摘要: 大型视觉语言模型（LVLM）现在构成了一个严重但被忽视的隐私威胁，因为它们可以直接从共享图像中推断社交媒体用户的地理位置，从而导致意外的隐私泄露。虽然对抗性图像扰动为地理隐私保护提供了一个潜在的方向，但它们需要相对强的失真才能有效对抗LVLM，而LVLM会显着降低视觉质量并降低图像的共享价值。为了克服这一限制，我们将印刷攻击确定为通过在视觉内容之外添加文本扩展来保护地理隐私的一个有希望的方向。我们进一步研究哪些文本语义可以有效扰乱地理位置推断，并设计一种两阶段、语义感知的印刷攻击，该攻击可以生成欺骗性文本以保护用户隐私。跨三个数据集的广泛实验表明，我们的方法显着降低了五种最先进的商业LVLM的地理位置预测准确性，建立了针对新出现的地理隐私威胁的实用且视觉保护策略。



## **15. SGuard-v1: Safety Guardrail for Large Language Models**

SGuard-v1：大型语言模型的安全保障 cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.

摘要: 我们介绍了SGuard-v1，这是一种适用于大型语言模型（LLM）的轻量级安全护栏，它包括两个专门的模型，用于检测有害内容并在人工智能对话设置中屏幕对抗性提示。第一个组件ContentLayer经过培训，能够根据MLCommons危险分类法识别LLM提示和响应中的安全风险，MLCommons危险分类法是人工智能信任和安全评估的综合框架。第二个组件JailbreakLayer是经过精心设计的课程培训的，该课程涵盖了集成的数据集和之前对抗提示工作的结果，涵盖60种主要攻击类型，同时减轻了错误不安全的分类。SGuard-v1构建在支持12种语言的2B参数Granite-3.3- 2B-Direct模型之上。我们从收集和合成的数据中策划了大约140万个训练实例，并对基本模型执行指令调优，根据其指定功能将策划的数据分布在两个组件之间。通过对公共和专有安全基准的广泛评估，SGuard-v1实现了最先进的安全性能，同时保持重量轻，从而减少了部署费用。SGuard-v1还通过提供多类别安全预测及其二进制置信度分数来提高下游使用的可解释性。我们根据Apache-2.0许可发布了SGuard-v1，以支持人工智能安全方面的进一步研究和实际部署。



## **16. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

GRAPHTEXTACK：对LLM增强型GNN的现实黑匣子节点注入攻击 cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.

摘要: 文本属性图（TAG）结合了结构和文本节点信息，在许多领域中都无处不在。最近的工作将大型语言模型（LLM）与图形神经网络（GNN）集成，以联合建模语义和结构，从而产生更通用和更富有表达力的模型，在TAG基准测试上实现最先进的性能。然而，这种集成引入了双重漏洞：GNN对结构性扰动敏感，而LLM衍生的功能容易受到提示注入和对抗性措辞的影响。虽然现有的对抗性攻击在很大程度上独立地扰乱结构或文本，但我们发现单模式攻击只会导致LLM增强的GNN的适度降级。此外，许多现有的攻击都假设不切实际的能力，例如白盒访问或直接修改图形数据。为了解决这些差距，我们提出了GRAPHTEXTACK，这是针对LLM增强型GNN的第一个黑匣子、多模式{，中毒}节点注入攻击。GRAPHTEXTACK注入具有精心设计的结构和语义的节点，以降低模型性能，在现实的威胁模型下运行，而不依赖模型内部或代理模型。为了在连接性和特征分配的组合性、不可微搜索空间中导航，GRAPHTEXTACK引入了一种新颖的进化优化框架，该框架具有多目标适应度函数，该函数平衡了局部预测中断和全局图影响。对五个数据集和两个最先进的LLM增强GNN模型的广泛实验表明，GRAPHTEXTACK的表现显着优于12个强基线。



## **17. QMA Complete Quantum-Enhanced Kyber: Provable Security Through CHSH Nonlocality**

QMA完全量子增强的Kyber：通过CHSH非定域性证明安全性 quant-ph

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12318v1) [paper-pdf](https://arxiv.org/pdf/2511.12318v1)

**Authors**: Ilias Cherkaoui, Indrakshi Dey

**Abstract**: Post-quantum cryptography (PQC) must secure large-scale communication systems against quantum adversaries where classical hardness alone is insufficient and purely quantum schemes remain impractical. Lattice-based key encapsulation mechanisms (KEMs) such as CRYSTALS-Kyber provide efficient quantum-resistant primitives but rely solely on computational hardness assumptions that are susceptible to hybrid classical-quantum attacks. To overcome this limitation, we introduce the first Clauser-Horne-Shimony-Holt (CHSH)-certified Kyber protocol, which embeds quantum non-locality verification directly within the key exchange phase. The proposed design integrates CHSH entanglement tests using Einstein-Podolsky-Rosen (EPR) pairs to yield measurable quantum advantage values exceeding classical correlation limits, thereby coupling information--theoretic quantum guarantees with lattice-based computational security. Formal reductions demonstrate that any polynomial-time adversary breaking the proposed KEM must either solve the Module Learning With Errors (Module-LWE) problem or a Quantum Merlin-Arthur (QMA)-complete instance of the 2-local Hamiltonian problem, under the standard complexity assumption QMA $\subset$ NP. The construction remains fully compatible with the Fujisaki-Okamoto (FO) transform, preserving chosen-ciphertext attack (CCA) security and Kyber's efficiency profile. The resulting CHSH-augmented Kyber scheme therefore establishes a mathematically rigorous, hybrid post-quantum framework that unifies lattice cryptography and quantum non-locality to achieve verifiable, composable, and forward-secure key agreement.

摘要: 后量子密码学（PQC）必须保护大规模通信系统免受量子对手的侵害，而仅靠经典硬度是不够的，而且纯粹的量子方案仍然不切实际。CRYSTALS-Kyber等基于网格的密钥封装机制（KEM）提供了高效的量子抵抗基元，但仅依赖于容易受到混合经典量子攻击的计算硬度假设。为了克服这一限制，我们引入了第一个Clauser-Horne-Shimony-Holt（CHSH）认证的Kyber协议，该协议将量子非局部验证直接嵌入到密钥交换阶段中。所提出的设计使用爱因斯坦-波多尔斯基-罗森（EPR）对集成了CHSH纠缠测试，以产生超出经典相关限制的可测量量子优势值，从而将信息理论量子保证与基于格的计算安全性相结合。形式约简表明，在标准复杂性假设QMA $\subset$ NP下，任何破坏提出的KEM的多项时间对手都必须解决带错误的模块学习（模块-LWE）问题或2-局部Hamilton问题的量子梅林-亚瑟（QMA）完全实例。该结构仍然与Fujisaki-Okamoto（FO）转换完全兼容，保留了选择密文攻击（PCA）安全性和Kyber的效率概况。因此，由此产生的CHSH增强Kyber方案建立了一个数学上严格的混合后量子框架，该框架将格子密码学和量子非局部性统一起来，以实现可验证、可组合且前向安全的密钥协议。



## **18. Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification**

使用联邦学习和基于嵌入的NLP分类的LLM保护隐私的即时注入检测 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12295v1) [paper-pdf](https://arxiv.org/pdf/2511.12295v1)

**Authors**: Hasini Jayathilaka

**Abstract**: Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.

摘要: 提示注入攻击是对大型语言模型（LLM）的一种新兴威胁，使恶意用户能够通过精心设计的输入来操纵输出。现有的检测方法通常需要集中即时数据，从而产生重大的隐私风险。本文提出了一种基于联邦学习和嵌入式分类的保护隐私的即时注入检测框架。良性和对抗提示的精心策划的数据集通过句子嵌入进行编码，并用于训练集中式和联邦式逻辑回归模型。联邦方法通过在客户端之间仅共享模型参数来保护隐私，同时实现与集中式训练相当的检测性能。结果表明，在不暴露原始数据的情况下，有效的即时注入检测是可行的，使其成为LLM联邦安全的首批探索之一。尽管该数据集规模有限，但研究结果建立了强有力的概念验证，并强调了构建安全和隐私感知的LLM系统的新方向。



## **19. Calibrated Adversarial Sampling: Multi-Armed Bandit-Guided Generalization Against Unforeseen Attacks**

校准对抗抽样：针对不可预见攻击的多臂盗贼引导的概括 cs.LG

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12265v1) [paper-pdf](https://arxiv.org/pdf/2511.12265v1)

**Authors**: Rui Wang, Zeming Wei, Xiyue Zhang, Meng Sun

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to various adversarial perturbations. To address the safety concerns arising from these vulnerabilities, adversarial training (AT) has emerged as one of the most effective paradigms for enhancing the robustness of DNNs. However, existing AT frameworks primarily focus on a single or a limited set of attack types, leaving DNNs still exposed to attack types that may be encountered in practice but not addressed during training. In this paper, we propose an efficient fine-tuning method called Calibrated Adversarial Sampling (CAS) to address these issues. From the optimization perspective within the multi-armed bandit framework, it dynamically designs rewards and balances exploration and exploitation by considering the dynamic and interdependent characteristics of multiple robustness dimensions. Experiments on benchmark datasets show that CAS achieves superior overall robustness while maintaining high clean accuracy, providing a new paradigm for robust generalization of DNNs.

摘要: 众所周知，深度神经网络（DNN）容易受到各种对抗性扰动的影响。为了解决这些漏洞引起的安全问题，对抗训练（AT）已经成为增强DNN鲁棒性的最有效范例之一。然而，现有的AT框架主要关注单个或有限的攻击类型，使得DNN仍然暴露于在实践中可能遇到但在训练期间未解决的攻击类型。在本文中，我们提出了一种称为校准对抗采样（CAS）的有效微调方法来解决这些问题。它从多武装强盗框架内的优化角度，考虑多个稳健性维度的动态和相互依赖的特征，动态设计奖励并平衡探索和利用。基准数据集的实验表明，CAS在保持高清晰准确性的同时实现了卓越的整体鲁棒性，为DNN的鲁棒概括提供了新的范式。



## **20. AlignTree: Efficient Defense Against LLM Jailbreak Attacks**

AlignTree：有效防御LLM越狱攻击 cs.LG

Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12217v1) [paper-pdf](https://arxiv.org/pdf/2511.12217v1)

**Authors**: Gil Goren, Shahar Katz, Lior Wolf

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.

摘要: 大型语言模型（LLM）很容易受到绕过安全指南并生成有害内容的对抗攻击。缓解这些漏洞需要强大且计算高效的防御机制。然而，现有的方法要么会产生很高的计算成本，要么依赖于易于规避的轻量级防御，这使得它们对于现实世界的基于LLM的系统来说不切实际。在这项工作中，我们引入了AlignTree防御，它增强了模型对齐，同时保持了最小的计算负担。AlignTree在生成期间监控LLM激活，并使用高效的随机森林分类器检测未对齐行为。该分类器对两个信号进行操作：（i）拒绝方向--在未对齐的提示时激活的线性表示，以及（ii）捕获与有害内容相关的非线性特征的基于支持器的信号。与之前的方法不同，AlignTree不需要额外的提示或辅助警卫模型。通过大量实验，我们展示了AlignTree在多个LLM和基准测试中的效率和稳健性。



## **21. MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization**

MPD-SGR：膜电位分布驱动的替代梯度正则化鲁棒脉冲神经网络 cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12199v1) [paper-pdf](https://arxiv.org/pdf/2511.12199v1)

**Authors**: Runhao Jiang, Chengzhi Jiang, Rui Yan, Huajin Tang

**Abstract**: The surrogate gradient (SG) method has shown significant promise in enhancing the performance of deep spiking neural networks (SNNs), but it also introduces vulnerabilities to adversarial attacks. Although spike coding strategies and neural dynamics parameters have been extensively studied for their impact on robustness, the critical role of gradient magnitude, which reflects the model's sensitivity to input perturbations, remains underexplored. In SNNs, the gradient magnitude is primarily determined by the interaction between the membrane potential distribution (MPD) and the SG function. In this study, we investigate the relationship between the MPD and SG and its implications for improving the robustness of SNNs. Our theoretical analysis reveals that reducing the proportion of membrane potential lying within the gradient-available range of the SG function effectively mitigates the sensitivity of SNNs to input perturbations. Building upon this insight, we propose a novel MPD-driven surrogate gradient regularization (MPD-SGR) method, which enhances robustness by explicitly regularizing the MPD based on its interaction with the SG function. Extensive experiments across multiple image classification benchmarks and diverse network architectures confirm that the MPD-SGR method significantly enhances the resilience of SNNs to adversarial perturbations and exhibits strong generalizability across diverse network configurations, SG function variants, and spike encoding schemes.

摘要: 代理梯度（SG）方法在增强深度尖峰神经网络（SNN）的性能方面表现出了巨大的希望，但它也引入了对抗性攻击的漏洞。尽管尖峰编码策略和神经动力学参数对鲁棒性的影响已被广泛研究，但反映模型对输入扰动敏感性的梯度幅度的关键作用仍然没有得到充分研究。在SNN中，梯度大小主要由膜势分布（CPD）和SG函数之间的相互作用决定。在这项研究中，我们研究了CPD和SG之间的关系及其对提高SNN稳健性的影响。我们的理论分析表明，减少位于SG函数的梯度可用范围内的膜势比例可以有效地降低SNN对输入扰动的敏感性。基于这一见解，我们提出了一种新型的CPD驱动的代理梯度正规化（MPD-SGR）方法，该方法通过根据其与SG函数的相互作用显式正规化MPD来增强鲁棒性。跨多个图像分类基准和不同网络架构的广泛实验证实，MPD-SGR方法显着增强了SNN对对抗性扰动的弹性，并在不同网络配置、SG函数变体和尖峰编码方案中表现出强大的通用性。



## **22. Rethinking Deep Alignment Through The Lens Of Incomplete Learning**

从不完全学习的角度重新思考深度对齐 cs.LG

AAAI'26

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12155v1) [paper-pdf](https://arxiv.org/pdf/2511.12155v1)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran

**Abstract**: Large language models exhibit systematic vulnerabilities to adversarial attacks despite extensive safety alignment. We provide a mechanistic analysis revealing that position-dependent gradient weakening during autoregressive training creates signal decay, leading to incomplete safety learning where safety training fails to transform model preferences in later response regions fully. We introduce base-favored tokens -- vocabulary elements where base models assign higher probability than aligned models -- as computational indicators of incomplete safety learning and develop a targeted completion method that addresses undertrained regions through adaptive penalties and hybrid teacher distillation. Experimental evaluation across Llama and Qwen model families demonstrates dramatic improvements in adversarial robustness, with 48--98% reductions in attack success rates while preserving general capabilities. These results establish both a mechanistic understanding and practical solutions for fundamental limitations in safety alignment methodologies.

摘要: 尽管进行了广泛的安全调整，大型语言模型仍表现出对对抗攻击的系统性漏洞。我们提供了一种机制分析，揭示了自回归训练期间与位置相关的梯度减弱会导致信号衰减，从而导致不完整的安全学习，即安全训练未能完全改变后期响应区域中的模型偏好。我们引入了基础偏好的代币（基础模型分配的概率比对齐模型更高的词汇元素）作为不完全安全学习的计算指标，并开发了一种有针对性的完成方法，通过自适应惩罚和混合教师提炼来解决训练不足的区域。Llama和Qwen模型家族的实验评估表明，对抗稳健性有了显着提高，攻击成功率降低了48- 98%，同时保留了一般能力。这些结果为安全对齐方法的基本局限性建立了机械性的理解和实用的解决方案。



## **23. AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models**

AttackVLA：视觉-语言-动作模型上的对抗性和后门攻击基准 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12149v1) [paper-pdf](https://arxiv.org/pdf/2511.12149v1)

**Authors**: Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Vision-Language-Action (VLA) models enable robots to interpret natural-language instructions and perform diverse tasks, yet their integration of perception, language, and control introduces new safety vulnerabilities. Despite growing interest in attacking such models, the effectiveness of existing techniques remains unclear due to the absence of a unified evaluation framework. One major issue is that differences in action tokenizers across VLA architectures hinder reproducibility and fair comparison. More importantly, most existing attacks have not been validated in real-world scenarios. To address these challenges, we propose AttackVLA, a unified framework that aligns with the VLA development lifecycle, covering data construction, model training, and inference. Within this framework, we implement a broad suite of attacks, including all existing attacks targeting VLAs and multiple adapted attacks originally developed for vision-language models, and evaluate them in both simulation and real-world settings. Our analysis of existing attacks reveals a critical gap: current methods tend to induce untargeted failures or static action states, leaving targeted attacks that drive VLAs to perform precise long-horizon action sequences largely unexplored. To fill this gap, we introduce BackdoorVLA, a targeted backdoor attack that compels a VLA to execute an attacker-specified long-horizon action sequence whenever a trigger is present. We evaluate BackdoorVLA in both simulated benchmarks and real-world robotic settings, achieving an average targeted success rate of 58.4% and reaching 100% on selected tasks. Our work provides a standardized framework for evaluating VLA vulnerabilities and demonstrates the potential for precise adversarial manipulation, motivating further research on securing VLA-based embodied systems.

摘要: 视觉-语言-动作（VLA）模型使机器人能够解释自然语言指令并执行各种任务，但它们对感知、语言和控制的集成引入了新的安全漏洞。尽管人们对攻击此类模型的兴趣越来越大，但由于缺乏统一的评估框架，现有技术的有效性仍然不清楚。一个主要问题是VLA架构中动作标记器的差异阻碍了可重复性和公平比较。更重要的是，大多数现有的攻击尚未在现实世界场景中得到验证。为了应对这些挑战，我们提出了AttackVLA，这是一个与VLA开发生命周期保持一致的统一框架，涵盖数据构建、模型训练和推理。在此框架内，我们实施了一系列广泛的攻击，包括所有针对VLA的现有攻击以及最初为视觉语言模型开发的多种改编攻击，并在模拟和现实环境中对其进行评估。我们对现有攻击的分析揭示了一个关键差距：当前的方法往往会引发无针对性的失败或静态动作状态，从而导致驱动VLA执行精确的长期动作序列的有针对性的攻击在很大程度上未被探索。为了填补这一空白，我们引入了BackdoorVLA，这是一种有针对性的后门攻击，它迫使VLA在出现触发器时执行攻击者指定的长期行动序列。我们在模拟基准测试和现实世界机器人环境中评估BackdoorVLA，实现了58.4%的平均目标成功率，并在选定任务中达到100%。我们的工作提供了一个用于评估VLA漏洞的标准化框架，并展示了精确对抗操纵的潜力，推动了对保护基于VLA的嵌入式系统的进一步研究。



## **24. Explainable Transformer-Based Email Phishing Classification with Adversarial Robustness**

具有对抗鲁棒性的可解释的基于转换器的电子邮件网络钓鱼分类 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12085v1) [paper-pdf](https://arxiv.org/pdf/2511.12085v1)

**Authors**: Sajad U P

**Abstract**: Phishing and related cyber threats are becoming more varied and technologically advanced. Among these, email-based phishing remains the most dominant and persistent threat. These attacks exploit human vulnerabilities to disseminate malware or gain unauthorized access to sensitive information. Deep learning (DL) models, particularly transformer-based models, have significantly enhanced phishing mitigation through their contextual understanding of language. However, some recent threats, specifically Artificial Intelligence (AI)-generated phishing attacks, are reducing the overall system resilience of phishing detectors. In response, adversarial training has shown promise against AI-generated phishing threats. This study presents a hybrid approach that uses DistilBERT, a smaller, faster, and lighter version of the BERT transformer model for email classification. Robustness against text-based adversarial perturbations is reinforced using Fast Gradient Method (FGM) adversarial training. Furthermore, the framework integrates the LIME Explainable AI (XAI) technique to enhance the transparency of the DistilBERT architecture. The framework also uses the Flan-T5-small language model from Hugging Face to generate plain-language security narrative explanations for end-users. This combined approach ensures precise phishing classification while providing easily understandable justifications for the model's decisions.

摘要: 网络钓鱼和相关网络威胁正变得更加多样化和技术先进。其中，基于电子邮件的网络钓鱼仍然是最主要、最持久的威胁。这些攻击利用人类漏洞传播恶意软件或未经授权访问敏感信息。深度学习（DL）模型，特别是基于转换器的模型，通过对语言的上下文理解显着增强了网络钓鱼缓解。然而，最近的一些威胁，特别是人工智能（AI）生成的网络钓鱼攻击，正在降低网络钓鱼检测器的整体系统弹性。作为回应，对抗性训练已经显示出对抗人工智能生成的网络钓鱼威胁的前景。这项研究提出了一种混合方法，使用DistilBERT，DistilBERT是用于电子邮件分类的BERT Transformer模型的更小、更快和更轻版本。使用快速梯度方法（FGM）对抗训练，增强了针对基于文本的对抗性扰动的鲁棒性。此外，该框架集成了LIME可解释人工智能（XAI）技术，以增强DistilBERT架构的透明度。该框架还使用Hugging Face的Flan-T5-small语言模型为最终用户生成通俗语言的安全叙述解释。这种组合方法确保了精确的网络钓鱼分类，同时为模型的决策提供易于理解的理由。



## **25. BackWeak: Backdooring Knowledge Distillation Simply with Weak Triggers and Fine-tuning**

BackWeak：简单地通过弱触发器和微调进行后门知识提炼 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12046v1) [paper-pdf](https://arxiv.org/pdf/2511.12046v1)

**Authors**: Shanmin Wang, Dongdong Zhao

**Abstract**: Knowledge Distillation (KD) is essential for compressing large models, yet relying on pre-trained "teacher" models downloaded from third-party repositories introduces serious security risks -- most notably backdoor attacks. Existing KD backdoor methods are typically complex and computationally intensive: they employ surrogate student models and simulated distillation to guarantee transferability, and they construct triggers in a way similar to universal adversarial perturbations (UAPs), which being not stealthy in magnitude, inherently exhibit strong adversarial behavior. This work questions whether such complexity is necessary and constructs stealthy "weak" triggers -- imperceptible perturbations that have negligible adversarial effect. We propose BackWeak, a simple, surrogate-free attack paradigm. BackWeak shows that a powerful backdoor can be implanted by simply fine-tuning a benign teacher with a weak trigger using a very small learning rate. We demonstrate that this delicate fine-tuning is sufficient to embed a backdoor that reliably transfers to diverse student architectures during a victim's standard distillation process, yielding high attack success rates. Extensive empirical evaluations on multiple datasets, model architectures, and KD methods show that BackWeak is efficient, simpler, and often more stealthy than previous elaborate approaches. This work calls on researchers studying KD backdoor attacks to pay particular attention to the trigger's stealthiness and its potential adversarial characteristics.

摘要: 知识蒸馏（KD）对于压缩大型模型至关重要，但依赖从第三方存储库下载的预先训练的“教师”模型会带来严重的安全风险--最引人注目的是后门攻击。现有的KD后门方法通常很复杂且计算密集型：它们采用代理学生模型和模拟蒸馏来保证可移植性，并且它们以类似于通用对抗性扰动（UPC）的方式构建触发器，其幅度并不隐蔽，本质上表现出强烈的对抗行为。这项工作质疑这种复杂性是否必要，并构建了隐形的“弱”触发器--具有可忽略不计的对抗影响的不可感知的扰动。我们提出BackWeak，一种简单的无代理攻击范式。BackWeak表明，只需用非常小的学习率对良性教师进行微调，具有弱触发，就可以植入强大的后门。我们证明，这种微妙的微调足以嵌入一个后门，该后门在受害者的标准提炼过程中可靠地转移到不同的学生架构，从而产生很高的攻击成功率。对多个数据集、模型架构和KD方法的广泛经验评估表明，BackWeak比之前的精心设计的方法高效、更简单，而且往往更隐蔽。这项工作呼吁研究KD后门攻击的研究人员特别关注触发器的隐蔽性及其潜在的对抗特征。



## **26. Robust Bidirectional Associative Memory via Regularization Inspired by the Subspace Rotation Algorithm**

受子空间旋转算法启发，通过正规化实现鲁棒双向联想记忆 cs.LG

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11902v1) [paper-pdf](https://arxiv.org/pdf/2511.11902v1)

**Authors**: Ci Lin, Tet Yeap, Iluju Kiringa, Biwei Zhang

**Abstract**: Bidirectional Associative Memory (BAM) trained with Bidirectional Backpropagation (B-BP) often suffers from poor robustness and high sensitivity to noise and adversarial attacks. To address these issues, we propose a novel gradient-free training algorithm, the Bidirectional Subspace Rotation Algorithm (B-SRA), which significantly improves the robustness and convergence behavior of BAM. Through comprehensive experiments, we identify two key principles -- orthogonal weight matrices (OWM) and gradient-pattern alignment (GPA) -- as central to enhancing the robustness of BAM. Motivated by these findings, we introduce new regularization strategies into B-BP, resulting in models with greatly improved resistance to corruption and adversarial perturbations. We further conduct an ablation study across different training strategies to determine the most robust configuration and evaluate BAM's performance under a variety of attack scenarios and memory capacities, including 50, 100, and 200 associative pairs. Among all methods, the SAME configuration, which integrates both OWM and GPA, achieves the strongest resilience. Overall, our results demonstrate that B-SRA and the proposed regularization strategies lead to substantially more robust associative memories and open new directions for building resilient neural architectures.

摘要: 使用双向反向传播（B-BP）训练的双向关联记忆（BAM）通常鲁棒性较差，对噪音和对抗性攻击的敏感性较高。为了解决这些问题，我们提出了一种新型的无梯度训练算法--双向子空间旋转算法（B-HRA），它显着提高了BAM的鲁棒性和收敛行为。通过全面的实验，我们确定了两个关键原则--垂直权重矩阵（OWM）和梯度模式对齐（GMA）--作为增强BAM鲁棒性的核心。受这些发现的激励，我们在B-BP中引入了新的正规化策略，使模型对腐败和对抗性扰动的抵抗力大大提高。我们进一步对不同的训练策略进行消融研究，以确定最稳健的配置并评估BAM在各种攻击场景和记忆容量（包括50、100和200个关联对）下的性能。在所有方法中，集成OWM和GMA的NPS配置实现了最强的弹性。总体而言，我们的结果表明，B-HRA和提出的正规化策略带来了更加稳健的联想记忆，并为构建弹性神经架构开辟了新的方向。



## **27. On the Trade-Off Between Transparency and Security in Adversarial Machine Learning**

对抗性机器学习中透明度和安全性之间的权衡 cs.LG

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11842v1) [paper-pdf](https://arxiv.org/pdf/2511.11842v1)

**Authors**: Lucas Fenaux, Christopher Srinivasa, Florian Kerschbaum

**Abstract**: Transparency and security are both central to Responsible AI, but they may conflict in adversarial settings. We investigate the strategic effect of transparency for agents through the lens of transferable adversarial example attacks. In transferable adversarial example attacks, attackers maliciously perturb their inputs using surrogate models to fool a defender's target model. These models can be defended or undefended, with both players having to decide which to use. Using a large-scale empirical evaluation of nine attacks across 181 models, we find that attackers are more successful when they match the defender's decision; hence, obscurity could be beneficial to the defender. With game theory, we analyze this trade-off between transparency and security by modeling this problem as both a Nash game and a Stackelberg game, and comparing the expected outcomes. Our analysis confirms that only knowing whether a defender's model is defended or not can sometimes be enough to damage its security. This result serves as an indicator of the general trade-off between transparency and security, suggesting that transparency in AI systems can be at odds with security. Beyond adversarial machine learning, our work illustrates how game-theoretic reasoning can uncover conflicts between transparency and security.

摘要: 透明度和安全性都是负责任人工智能的核心，但它们在敌对环境中可能会发生冲突。我们通过可转移对抗性示例攻击的视角来研究透明度对代理人的战略影响。在可转移的对抗示例攻击中，攻击者使用代理模型恶意扰乱他们的输入，以欺骗防御者的目标模型。这些模型可以防御或不防御，双方玩家都必须决定使用哪一个。通过对181个模型中的9种攻击进行大规模实证评估，我们发现当攻击者与防御者的决定相匹配时，他们会更成功;因此，默默无闻可能对防御者有利。利用博弈论，我们通过将这个问题建模为纳什博弈和斯塔克伯格博弈，并比较预期结果，来分析透明度和安全性之间的权衡。我们的分析证实，只有知道防御者的模型是否受到防御，有时就足以损害其安全性。这一结果是透明度和安全性之间总体权衡的指标，表明人工智能系统的透明度可能与安全性不一致。除了对抗性机器学习之外，我们的工作还说明了博弈论推理如何揭示透明度和安全性之间的冲突。



## **28. Incentive Attacks in BTC: Short-Term Revenue Changes and Long-Term Efficiencies**

BTC中的激励攻击：短期收入变化和长期效应 cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11538v1) [paper-pdf](https://arxiv.org/pdf/2511.11538v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Bitcoin's (BTC) Difficulty Adjustment Algorithm (DAA) has been a source of vulnerability for incentive attacks such as selfish mining, block withholding and coin hopping strategies. In this paper, first, we rigorously study the short-term revenue change per hashpower of the adversarial and honest miners for these incentive attacks. To study the long-term effects, we introduce a new efficiency metric defined as the revenue/cost per hashpower per time for the attacker and the honest miners.   Our results indicate that the short-term benefits of intermittent mining strategies are negligible compared to the original selfish mining attack, and in the long-term, selfish mining provides better efficiency. We further demonstrate that a coin hopping strategy between BTC and Bitcoin Cash (BCH) relying on BTC DAA benefits the loyal honest miners of BTC in the same way and to the same extent per unit of computational power as it does the hopper in the short-term. For the long-term, we establish a new boundary between the selfish mining and coin hopping attack, identifying the optimal efficient strategy for each parameter.   For block withholding strategies, it turns out, the honest miners outside the pool profit from the attack, usually even more than the attacker both in the short-term and the long-term. Moreover, a power adjusting withholding attacker does not necessarily observe a profit lag in the short-term. It has been long thought that the profit lag of selfish mining is among the main reasons why such an attack has not been observed in practice. We show that such a barrier does not apply to power adjusting attacks and relatively small pools are at an immediate threat.

摘要: 比特币（BTC）的难度调整算法（DAA）一直是激励攻击的漏洞来源，如自私采矿，区块扣留和硬币跳跃策略。在本文中，首先，我们严格研究了这些激励攻击的对抗性和诚实矿工的短期收入变化。为了研究长期影响，我们引入了一个新的效率指标，定义为攻击者和诚实矿工每次每哈希功率的收入/成本。   我们的研究结果表明，间歇性挖掘策略的短期利益是可以忽略不计相比，原来的自私挖掘攻击，从长远来看，自私挖掘提供了更好的效率。我们进一步证明，依赖BTC DAA的BTC和比特币现金（BH）之间的硬币跳跃策略，可以以相同的方式和相同的程度，使BTC的忠实诚实矿工受益，单位计算能力与短期内的跳跃者受益。从长远来看，我们在自私挖掘和跳币攻击之间建立了新的边界，为每个参数确定最佳有效策略。   事实证明，对于区块扣留策略，池外诚实的矿工从攻击中获利，无论是短期还是长期，通常都比攻击者更多。此外，权力调整预扣税攻击者不一定会观察到短期内的利润滞后。长期以来，人们一直认为自私采矿的利润滞后是实践中未观察到此类攻击的主要原因之一。我们表明，这样的障碍不适用于功率调整攻击，相对较小的池立即受到威胁。



## **29. Class-feature Watermark: A Resilient Black-box Watermark Against Model Extraction Attacks**

类特征水印：一种抗模型抽取攻击的弹性黑箱水印 cs.CR

Accepted by AAAI'26

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.07947v2) [paper-pdf](https://arxiv.org/pdf/2511.07947v2)

**Authors**: Yaxin Xiao, Qingqing Ye, Zi Liang, Haoyang Li, RongHua Li, Huadi Zheng, Haibo Hu

**Abstract**: Machine learning models constitute valuable intellectual property, yet remain vulnerable to model extraction attacks (MEA), where adversaries replicate their functionality through black-box queries. Model watermarking counters MEAs by embedding forensic markers for ownership verification. Current black-box watermarks prioritize MEA survival through representation entanglement, yet inadequately explore resilience against sequential MEAs and removal attacks. Our study reveals that this risk is underestimated because existing removal methods are weakened by entanglement. To address this gap, we propose Watermark Removal attacK (WRK), which circumvents entanglement constraints by exploiting decision boundaries shaped by prevailing sample-level watermark artifacts. WRK effectively reduces watermark success rates by at least 88.79% across existing watermarking benchmarks.   For robust protection, we propose Class-Feature Watermarks (CFW), which improve resilience by leveraging class-level artifacts. CFW constructs a synthetic class using out-of-domain samples, eliminating vulnerable decision boundaries between original domain samples and their artifact-modified counterparts (watermark samples). CFW concurrently optimizes both MEA transferability and post-MEA stability. Experiments across multiple domains show that CFW consistently outperforms prior methods in resilience, maintaining a watermark success rate of at least 70.15% in extracted models even under the combined MEA and WRK distortion, while preserving the utility of protected models.

摘要: 机器学习模型构成宝贵的知识产权，但仍然容易受到模型提取攻击（EMA）的影响，即对手通过黑匣子查询复制其功能。模型水印通过嵌入取证标记进行所有权验证来对抗多边环境协定。当前的黑匣子水印通过表示纠缠优先考虑多边环境协定的生存，但没有充分探索针对连续多边环境协定和删除攻击的弹性。我们的研究表明，这种风险被低估了，因为现有的去除方法被纠缠削弱了。为了解决这一差距，我们提出了水印去除attacK（WRK），它通过利用由流行的样本级水印伪影塑造的决策边界来规避纠缠限制。WRK在现有水印基准中有效地降低了至少88.79%的水印成功率。   为了强大的保护，我们提出了类特征水印（CFW），它通过利用类级工件来提高弹性。CFW使用域外样本构建合成类，消除原始域样本与其经伪影修改的对应样本（水印样本）之间的脆弱决策边界。CFW同时优化MEA的可转移性和MEA后的稳定性。多个领域的实验表明，CFW在弹性方面始终优于先前的方法，即使在MEA和WRK组合失真的情况下，提取的模型仍保持至少70.15%的水印成功率，同时保留了受保护模型的实用性。



## **30. A Generative Adversarial Approach to Adversarial Attacks Guided by Contrastive Language-Image Pre-trained Model**

对比图像预训练模型引导的对抗攻击生成性方法 cs.CV

18 pages, 3 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.01317v2) [paper-pdf](https://arxiv.org/pdf/2511.01317v2)

**Authors**: Sampriti Soor, Alik Pramanick, Jothiprakash K, Arijit Sur

**Abstract**: The rapid growth of deep learning has brought about powerful models that can handle various tasks, like identifying images and understanding language. However, adversarial attacks, an unnoticed alteration, can deceive models, leading to inaccurate predictions. In this paper, a generative adversarial attack method is proposed that uses the CLIP model to create highly effective and visually imperceptible adversarial perturbations. The CLIP model's ability to align text and image representation helps incorporate natural language semantics with a guided loss to generate effective adversarial examples that look identical to the original inputs. This integration allows extensive scene manipulation, creating perturbations in multi-object environments specifically designed to deceive multilabel classifiers. Our approach integrates the concentrated perturbation strategy from Saliency-based Auto-Encoder (SSAE) with the dissimilar text embeddings similar to Generative Adversarial Multi-Object Scene Attacks (GAMA), resulting in perturbations that both deceive classification models and maintain high structural similarity to the original images. The model was tested on various tasks across diverse black-box victim models. The experimental results show that our method performs competitively, achieving comparable or superior results to existing techniques, while preserving greater visual fidelity.

摘要: 深度学习的快速发展带来了强大的模型，可以处理各种任务，例如识别图像和理解语言。然而，对抗性攻击（一种未被注意到的改变）可能会欺骗模型，导致预测不准确。本文提出了一种生成式对抗攻击方法，该方法使用CLIP模型来创建高效且视觉上不可感知的对抗扰动。CLIP模型对齐文本和图像表示的能力有助于将自然语言语义与引导损失结合起来，以生成看起来与原始输入相同的有效对抗示例。这种集成允许广泛的场景操作，在多对象环境中创建扰动，专门设计用于欺骗多标签分类器。我们的方法将基于显著性的自动编码器（SSAE）的集中扰动策略与类似于生成式对抗多对象场景攻击（GAMA）的不同文本嵌入相结合，从而产生既欺骗分类模型又保持与原始图像高度结构相似性的扰动。该模型在各种黑盒受害者模型的各种任务上进行了测试。实验结果表明，我们的方法具有竞争力，实现可比或优于现有技术的结果，同时保持更大的视觉保真度。



## **31. CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents**

CompressionAttack：利用即时压缩作为LLM支持的代理中的新攻击表面 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.22963v3) [paper-pdf](https://arxiv.org/pdf/2510.22963v3)

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to an average ASR of 83% and 87% in two tasks, while remaining highly stealthy and transferable. Case studies in three practical scenarios confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.

摘要: LLM支持的代理通常使用即时压缩来降低推理成本，但这会带来新的安全风险。压缩模块针对效率而不是安全性进行了优化，可以通过对抗输入来操纵，从而导致语义漂移并改变LLM行为。这项工作将即时压缩确定为一种新型攻击表面，并提出了第一个利用它的框架CompressionAttack。CompressionAttack包括两种策略：HardCom，使用离散对抗编辑进行硬压缩，以及SoftCom，为软压缩执行潜伏空间扰动。多个LLM的实验显示，两项任务的平均ZR分别高达83%和87%，同时保持高度隐蔽性和可转移性。三种实际场景中的案例研究证实了现实世界的影响，而当前的防御措施被证明无效，凸显了加强保护的必要性。



## **32. A Lightweight Approach for State Machine Replication**

状态机复制的轻量级方法 cs.DC

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.17771v2) [paper-pdf](https://arxiv.org/pdf/2509.17771v2)

**Authors**: Christian Cachin, Jinfeng Dou, Christian Scheideler, Philipp Schneider

**Abstract**: We present a lightweight solution for state machine replication with commitment certificates. Specifically, we adapt and analyze a median rule for the stabilizing consensus problem [Doerr11] to operate in a client-server setting where arbitrary servers may be blocked adaptively based on past system information. We further extend our protocol by compressing information about committed commands, thus keeping the protocol lightweight, while still enabling clients to easily prove that their commands have indeed been committed on the shared state. Our approach guarantees liveness as long as at most a constant fraction of servers are blocked, ensures safety under any number of blocked servers, and supports fast recovery even after all servers are blocked. In addition to offering near-optimal asymptotic performance in several respects, our method is fully decentralized, unlike other near-optimal solutions that rely on leaders. In particular, our solution is robust against adversaries that target key servers (which captures insider-based denial-of-service attacks), whereas leader-based approaches fail under such a blocking model.

摘要: 我们提供了一种用于具有承诺证书的状态机复制的轻量级解决方案。具体来说，我们调整并分析稳定共识问题[Doerr 11]的中间规则，以在客户端-服务器设置中操作，其中可以根据过去的系统信息自适应地阻止任意服务器。我们通过压缩有关已提交命令的信息来进一步扩展我们的协议，从而保持协议轻量级，同时仍然使客户端能够轻松证明他们的命令确实已在共享状态上提交。只要最多有一定比例的服务器被阻止，我们的方法就能保证活跃性，确保任何数量被阻止的服务器下的安全性，并且即使在所有服务器被阻止后也支持快速恢复。除了在几个方面提供接近最优的渐进性能外，我们的方法还完全去中心化，与其他依赖领导者的接近最优的解决方案不同。特别是，我们的解决方案对于针对关键服务器（捕获基于内部的拒绝服务攻击）的对手来说是强大的，而基于领导者的方法在这种拦截模型下会失败。



## **33. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **34. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

SoK：传感器攻击如何扰乱自动驾驶车辆：端到端分析、挑战和错过的威胁 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11120v4) [paper-pdf](https://arxiv.org/pdf/2509.11120v4)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self driving cars, ground robots, and drones, rely on multi-modal sensor pipelines for safe operation, yet remain vulnerable to adversarial sensor attacks. A critical gap is the lack of a systematic end-to-end view of how sensor induced errors traverse interconnected modules to affect the physical world. To bridge the gap, we provide a comprehensive survey across platforms, sensing modalities, attack methods, and countermeasures. At its core is \Model (\modelAbbr), a graph-based illustrative framework that maps how attacks inject errors, the conditions for their propagation through modules from perception and localization to planning and control, and when they reach physical impact. From the systematic analysis, our study distills 8 key findings that highlight the feasibility challenges of sensor attacks and uncovers 12 previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments.

摘要: 包括自动驾驶汽车、地面机器人和无人机在内的自动驾驶车辆依赖多模态传感器管道来实现安全操作，但仍然容易受到对抗性传感器攻击。一个关键的差距是缺乏一个系统的端到端的观点，传感器引起的错误如何穿越互连的模块，影响物理世界。为了弥补这一差距，我们提供了一个跨平台，传感模式，攻击方法和对策的全面调查。其核心是\Model（\modelAbbr），这是一个基于图形的说明性框架，它映射了攻击如何注入错误，它们通过模块从感知和定位到规划和控制的传播条件，以及它们何时达到物理影响。从系统分析中，我们的研究提炼出8个关键发现，这些发现凸显了传感器攻击的可行性挑战，并揭示了12种以前被忽视的利用模块间交互的攻击载体，其中一些我们通过概念验证实验进行了验证。



## **35. Isolate Trigger: Detecting and Eliminating Adaptive Backdoor Attacks**

隔离触发：检测和消除自适应后门攻击 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2508.04094v2) [paper-pdf](https://arxiv.org/pdf/2508.04094v2)

**Authors**: Chengrui Sun, Hua Zhang, Haoran Gao, Shang Wang, Zian Tian, Jianjin Zhao, Qi Li, Hongliang Zhu, Zongliang Shen, Anmin Fu

**Abstract**: Deep learning models are widely deployed in various applications but remain vulnerable to stealthy adversarial threats, particularly backdoor attacks. Backdoor models trained on poisoned datasets behave normally with clean inputs but cause mispredictions when a specific trigger is present. Most existing backdoor defenses assume that adversaries only inject one backdoor with small and conspicuous triggers. However, adaptive backdoor that entangle multiple trigger patterns with benign features can effectively bypass existing defenses. To defend against these attacks, we propose Isolate Trigger (IsTr), an accurate and efficient framework for backdoor detection and mitigation. IsTr aims to eliminate the influence of benign features and reverse hidden triggers. IsTr is motivated by the observation that a model's feature extractor focuses more on benign features while its classifier focuses more on trigger patterns. Based on this difference, IsTr designs Steps and Differential-Middle-Slice to resolve the detecting challenge of isolating triggers from benign features. Moreover, IsTr employs unlearning-based repair to remove both attacker-injected and natural backdoors while maintaining model benign accuracy. We extensively evaluate IsTr against six representative backdoor attacks and compare with seven state-of-the-art baseline methods across three real-world applications: digit recognition, face recognition, and traffic sign recognition. In most cases, IsTr reduces detection overhead by an order of magnitude while achieving over 95\% detection accuracy and maintaining the post-repair attack success rate below 3\%, outperforming baseline defenses. IsTr remains robust against various adaptive attacks, even when trigger patterns are heavily entangled with benign features.

摘要: 深度学习模型广泛部署在各种应用程序中，但仍然容易受到隐形对抗威胁，尤其是后门攻击。在有毒数据集上训练的后门模型在干净的输入下表现正常，但当存在特定触发器时会导致预测错误。大多数现有的后门防御都假设对手只用小而明显的触发器注入一个后门。然而，将多个触发模式与良性特征纠缠在一起的自适应后门可以有效地绕过现有的防御。为了抵御这些攻击，我们提出了Isolate Trigger（Isrol），这是一个准确有效的后门检测和缓解框架。IsTR旨在消除良性特征的影响并逆转隐藏触发因素。IsTR的动机是这样一个观察：模型的特征提取器更关注良性特征，而其分类器更关注触发模式。基于这种差异，IsTR设计了Steps和Differential-Middle-Slice来解决将触发器与良性特征隔离的检测挑战。此外，IsTR采用基于无学习的修复来删除攻击者注入的后门和自然后门，同时保持模型的良性准确性。我们针对六种有代表性的后门攻击进行了广泛评估，并与三种现实应用程序中的七种最先进的基线方法进行了比较：数字识别、面部识别和交通标志识别。在大多数情况下，IsTLR将检测费用降低了一个数量级，同时实现超过95%的检测准确率，并将修复后攻击成功率保持在3%以下，优于基线防御。即使触发模式与良性特征严重纠缠在一起，IsTLR仍然能够抵御各种自适应攻击。



## **36. LeakyCLIP: Extracting Training Data from CLIP**

LeakyCLIP：从CLIP中提取训练数据 cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.00756v3) [paper-pdf](https://arxiv.org/pdf/2508.00756v3)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 258% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.

摘要: 了解对比语言-图像预训练（CLIP）中的记忆和隐私泄露风险对于确保多模式模型的安全性至关重要。最近的研究证明了从扩散模型中提取敏感训练示例的可行性，条件扩散模型表现出更强的记忆和泄露信息的倾向。在这项工作中，我们通过CLIP倒置的镜头调查了CLIP中的数据记忆和提取风险，这是一个旨在根据文本提示重建训练图像的过程。为此，我们引入了\textBF{LeakyCLIP}，这是一种新型攻击框架，旨在从CLIP嵌入中实现高质量、语义准确的图像重建。我们确定了CLIP倒置中的三个关键挑战：1）非鲁棒特征，2）文本嵌入中的视觉语义有限，以及3）重建保真度低。为了解决这些挑战，LeakyCLIP采用1）对抗性微调以增强优化平滑度，2）基于线性变换的嵌入对齐，以及3）基于稳定扩散的细化以提高保真度。经验结果证明了LeakyCLIP的优越性，与LAION-2B子集的基线方法相比，ViT-B-16的结构相似性指数测量（SSIM）提高了258%以上。此外，我们还发现了普遍存在的泄露风险，表明训练数据成员关系甚至可以从低保真重建的指标中成功推断出来。我们的工作介绍了一种实用的CLIP倒置方法，同时为多模式模型中隐私风险的性质和范围提供了新颖的见解。



## **37. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2507.22564v2) [paper-pdf](https://arxiv.org/pdf/2507.22564v2)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **38. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2506.09443v2) [paper-pdf](https://arxiv.org/pdf/2506.09443v2)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.

摘要: 大型语言模型（LLM）在不同任务中表现出了卓越的能力，推动了LLM作为法官自动评估系统的开发和广泛采用，包括红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发对其稳健性和可信性的严重担忧。基于LLM的法官的现有评估方法往往支离破碎，缺乏全面稳健性评估的统一框架。此外，人们很少探讨即时模板设计和模型选择对判断稳健性的影响，而且它们在现实世界部署中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。具体来说，RobustJudge调查了12个模型中15种攻击方法和7种防御策略的有效性（RJ 1），检查了即时模板设计和模型选择的影响（RJ 2），并评估现实世界部署的安全性（RJ 3）。我们的研究得出了三个关键发现：（1）LLM as-a-Judge系统极易受到PAIR和组合攻击等攻击，而重标记化和基于LLM的检测器等防御机制可以提供增强的保护;（2）不同提示模板的稳健性差异很大（高达40%）;（3）在阿里巴巴的PRI平台上部署RobustJudge发现了之前未发现的漏洞。这些结果为构建值得信赖的法学硕士作为法官系统提供了实用见解。



## **39. Towards Cross-Domain Multi-Targeted Adversarial Attacks**

走向跨领域多目标对抗攻击 cs.CV

Under review

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.20782v2) [paper-pdf](https://arxiv.org/pdf/2505.20782v2)

**Authors**: Taïga Gonçalves, Tomo Miyazaki, Shinichiro Omachi

**Abstract**: Multi-targeted adversarial attacks aim to mislead classifiers toward specific target classes using a single perturbation generator with a conditional input specifying the desired target class. Existing methods face two key limitations: (1) a single generator supports only a limited number of predefined target classes, and (2) it requires access to the victim model's training data to learn target class semantics. This dependency raises data leakage concerns in practical black-box scenarios where the training data is typically private. To address these limitations, we propose a novel Cross-Domain Multi-Targeted Attack (CD-MTA) that can generate perturbations toward arbitrary target classes, even those that do not exist in the attacker's training data. CD-MTA is trained on a single public dataset but can perform targeted attacks on black-box models trained on different datasets with disjoint and unknown class sets. Our method requires only a single example image that visually represents the desired target class, without relying its label, class distribution or pretrained embeddings. We achieve this through a Feature Injection Module (FIM) and class-agnostic objectives which guide the generator to extract transferable, fine-grained features from the target image without inferring class semantics. Experiments on ImageNet and seven additional datasets show that CD-MTA outperforms existing multi-targeted attack methods on unseen target classes in black-box and cross-domain scenarios. The code is available at https://github.com/tgoncalv/CD-MTA.

摘要: 多目标对抗攻击旨在使用单个扰动生成器，该生成器具有指定所需目标类别的条件输入，将分类器误导至特定目标类别。现有的方法面临两个关键限制：（1）单个生成器仅支持有限数量的预定义目标类，以及（2）它需要访问受害者模型的训练数据才能学习目标类语义。这种依赖性在训练数据通常是私人的实际黑匣子场景中引发了数据泄露问题。为了解决这些限制，我们提出了一种新型的跨域多目标攻击（CD-MTA），它可以对任意目标类别产生扰动，甚至是那些在攻击者的训练数据中不存在的扰动。CD-MTA在单个公共数据集上训练，但可以对在具有不相交且未知类集的不同数据集上训练的黑匣子模型执行有针对性的攻击。我们的方法只需要一个直观地表示所需目标类的单个示例图像，而不依赖其标签、类分布或预训练嵌入。我们通过特征注入模块（RST）和类不可知目标来实现这一目标，这些目标引导生成器从目标图像中提取可转移的细粒度特征，而无需推断类语义。ImageNet和其他七个数据集的实验表明，在黑匣子和跨域场景中，CD-MTA优于现有的针对未见目标类的多目标攻击方法。该代码可在https://github.com/tgoncalv/CD-MTA上获取。



## **40. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

SafeKey：放大啊哈时刻洞察以实现安全推理 cs.AI

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.16186v2) [paper-pdf](https://arxiv.org/pdf/2505.16186v2)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.

摘要: 大型推理模型（LRM）引入了新一代在回答之前进行显式推理的范式，导致复杂任务的显着改进。然而，它们对有害查询和对抗性攻击构成了巨大的安全风险。虽然最近针对LRM的主流安全工作（监督微调（SFT））提高了安全性能，但我们发现与SFT一致的模型很难推广到看不见的越狱提示。在对LRM一代进行彻底调查后，我们发现了一个可以激活安全推理并导致安全响应的安全啊哈时刻。这个啊哈时刻通常出现在“关键时刻”中，它遵循模型的查询理解过程，并且可以指示模型是否将安全地继续进行。基于这些见解，我们提出了SafeKey，其中包括两个补充目标，以更好地激活关键句中的安全啊哈时刻：（1）双路径安全头，以增强关键句之前模型内部表示中的安全信号，（2）查询面具建模目标，以提高模型对其查询理解的关注度，这具有重要的安全提示。跨多个安全基准的实验表明，我们的方法显着提高了对各种越狱攻击和分发外有害提示的安全概括，将平均危害率降低9.6%，同时保持一般能力。我们的分析揭示了SafeKey如何通过重塑内部注意力和提高隐藏表示的质量来增强安全性。



## **41. Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models**

稳健语言模型的思想链驱动的对抗场景外推 cs.CL

19 pages, 5 figures. Accepted in AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.17089v2) [paper-pdf](https://arxiv.org/pdf/2505.17089v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.

摘要: 大型语言模型（LLM）表现出令人印象深刻的能力，但仍然容易受到越来越多的安全风险的影响，包括越狱、有毒内容、幻觉和偏见。现有的防御系统通常只解决单一威胁类型，或者诉诸严格的彻底拒绝，牺牲用户体验，并且未能普遍适用于各种新颖的攻击。本文介绍了对抗场景外推（ASE），这是一种新型的推理时计算框架，它利用思想链（CoT）推理来同时增强LLM稳健性和无缝性。ASE指导LLM完成一个自我生成的过程，即在对用户查询做出响应之前考虑潜在的对抗场景并制定防御策略。对四种最新LLM的四种对抗基准进行的综合评估表明，ASE的越狱攻击成功率接近零，毒性最小，同时将彻底拒绝率降至< 4%。ASE在稳健性与无缝性权衡方面优于六种最先进的防御，对抗性问答的准确率为92-99%，偏见得分低4- 10倍。通过将对抗性感知转化为内在认知过程，ASE为安全、自然的人机交互设定了新范式。



## **42. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

利用ViT中的计算冗余来提高对抗性可移植性 cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2504.10804v3) [paper-pdf](https://arxiv.org/pdf/2504.10804v3)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.

摘要: Vision Transformers（ViT）在一系列应用中表现出令人印象深刻的性能，包括许多安全关键任务。然而，它们独特的架构属性在对抗稳健性方面提出了新的挑战和机遇。特别是，我们观察到，与CNN上制作的对抗示例相比，在ViT上制作的对抗示例表现出更高的可转移性，这表明ViT包含有利于转移攻击的结构特征。在这项工作中，我们研究了计算冗余在ViT中的作用及其对对抗可转移性的影响。与之前旨在减少计算以提高效率的研究不同，我们建议利用这种冗余来提高对抗性示例的质量和可移植性。通过详细的分析，我们确定了两种形式的冗余，包括数据级和模型级，可以利用它们来放大攻击有效性。基于这一见解，我们设计了一套技术，包括注意力稀疏性操纵、注意力头排列、干净令牌正规化、幽灵MoE多样化和测试时对抗训练。ImageNet-1 k数据集上的大量实验验证了我们方法的有效性，表明我们的方法在跨不同模型架构的可移植性和通用性方面都显着优于现有基线。



## **43. Backdooring CLIP through Concept Confusion**

通过概念混乱为CLIP做后门 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2503.09095v2) [paper-pdf](https://arxiv.org/pdf/2503.09095v2)

**Authors**: Lijie Hu, Junchi Liao, Weimin Lyu, Shaopeng Fu, Tianhao Huang, Shu Yang, Guimin Hu, Di Wang

**Abstract**: Backdoor attacks pose a serious threat to deep learning models by allowing adversaries to implant hidden behaviors that remain dormant on clean inputs but are maliciously triggered at inference. Existing backdoor attack methods typically rely on explicit triggers such as image patches or pixel perturbations, which makes them easier to detect and limits their applicability in complex settings. To address this limitation, we take a different perspective by analyzing backdoor attacks through the lens of concept-level reasoning, drawing on insights from interpretable AI. We show that traditional attacks can be viewed as implicitly manipulating the concepts activated within a model's latent space. This motivates a natural question: can backdoors be built by directly manipulating concepts? To answer this, we propose the Concept Confusion Attack (CCA), a novel framework that designates human-understandable concepts as internal triggers, eliminating the need for explicit input modifications. By relabeling images that strongly exhibit a chosen concept and fine-tuning on this mixed dataset, CCA teaches the model to associate the concept itself with the attacker's target label. Consequently, the presence of the concept alone is sufficient to activate the backdoor, making the attack stealthier and more resistant to existing defenses. Using CLIP as a case study, we show that CCA achieves high attack success rates while preserving clean-task accuracy and evading state-of-the-art defenses.

摘要: 后门攻击允许对手植入隐藏行为，这些行为在干净的输入上保持休眠状态，但在推理时被恶意触发，从而对深度学习模型构成严重威胁。现有的后门攻击方法通常依赖于显式触发，例如图像补丁或像素扰动，这使得它们更容易检测并限制了它们在复杂设置中的适用性。为了解决这一局限性，我们采取了不同的角度，通过概念级推理的视角分析后门攻击，借鉴可解释人工智能的见解。我们表明，传统的攻击可以被视为隐含地操纵模型潜在空间内激活的概念。这引发了一个自然的问题：后门可以通过直接操纵概念来构建吗？为了解决这个问题，我们提出了概念混淆攻击（PCA），这是一种新颖的框架，将人类可理解的概念指定为内部触发器，消除了对显式输入修改的需要。通过重新标记强烈表现出所选概念的图像并对这个混合数据集进行微调，PCA教模型将概念本身与攻击者的目标标签关联起来。因此，仅该概念的存在就足以激活后门，使攻击更隐蔽，对现有防御更具抵抗力。使用CLIP作为案例研究，我们表明，CAA在保持清晰任务准确性并规避最先进的防御的同时实现了很高的攻击成功率。



## **44. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: OpenAI-o3和DeepSeek-R1等大型推理模型（LRM）的快速发展使复杂推理相对于非推理大型语言模型（LRM）有了显着改进。然而，它们增强的功能，加上DeepSeek-R1等模型的开源访问，引发了严重的安全问题，特别是关于它们被滥用的可能性。在这项工作中，我们对这些推理模型进行了全面的安全评估，利用既定的安全基准来评估它们对安全法规的遵守性。此外，我们还调查了它们对越狱和即时注射等对抗攻击的敏感性，以评估它们在现实应用中的稳健性。通过多方面的分析，我们发现了四个关键发现：（1）开源推理模型和o3-mini模型之间在安全基准和攻击方面存在显着的安全差距，这表明需要对开放LRM做出更多的安全努力。(2)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(3)安全思维出现在LRM的推理过程中，但在对抗性攻击时经常失败。(4)R1模型中的思维过程比其最终答案带来了更大的安全问题。我们的研究为推理模型的安全性影响提供了深入的见解，并强调了进一步提高R1模型安全性以缩小差距的必要性。



## **45. MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework**

MOS攻击：可扩展的多目标对抗攻击框架 cs.LG

Camera ready version of CVPR 2025

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2501.07251v3) [paper-pdf](https://arxiv.org/pdf/2501.07251v3)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang

**Abstract**: Crafting adversarial examples is crucial for evaluating and enhancing the robustness of Deep Neural Networks (DNNs), presenting a challenge equivalent to maximizing a non-differentiable 0-1 loss function.   However, existing single objective methods, namely adversarial attacks focus on a surrogate loss function, do not fully harness the benefits of engaging multiple loss functions, as a result of insufficient understanding of their synergistic and conflicting nature.   To overcome these limitations, we propose the Multi-Objective Set-based Attack (MOS Attack), a novel adversarial attack framework leveraging multiple loss functions and automatically uncovering their interrelations.   The MOS Attack adopts a set-based multi-objective optimization strategy, enabling the incorporation of numerous loss functions without additional parameters.   It also automatically mines synergistic patterns among various losses, facilitating the generation of potent adversarial attacks with fewer objectives.   Extensive experiments have shown that our MOS Attack outperforms single-objective attacks. Furthermore, by harnessing the identified synergistic patterns, MOS Attack continues to show superior results with a reduced number of loss functions. Our code is available at https://github.com/pgg3/MOS-Attack.

摘要: 制作对抗性示例对于评估和增强深度神经网络（DNN）的稳健性至关重要，这带来了相当于最大化不可微0-1损失函数的挑战。   然而，现有的单目标方法，即专注于替代损失函数的对抗性攻击，并没有充分利用参与多个损失函数的好处，因为对其协同和冲突性质的理解不足。   为了克服这些限制，我们提出了基于多目标集的攻击（MOS攻击），这是一种新型的对抗攻击框架，利用多个损失函数并自动揭示它们的相互关系。   MOS Attack采用基于集合的多目标优化策略，无需额外参数即可合并众多损失函数。   它还自动挖掘各种损失之间的协同模式，促进生成目标更少的强大对抗性攻击。   大量的实验表明，我们的MOS攻击优于单目标攻击。此外，通过利用所识别的协同模式，MOS攻击继续显示出优异的结果，减少了损失函数的数量。我们的代码可以在https://github.com/pgg3/MOS-Attack上找到。



## **46. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

基于视频的MLLM中对抗性攻击的可转移性：跨模式图像到视频的方法 cs.CV

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2501.01042v3) [paper-pdf](https://arxiv.org/pdf/2501.01042v3)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.

摘要: 基于视频的多模式大型语言模型（V-MLLM）在视频-文本多模式任务中表现出对对抗示例的脆弱性。然而，对抗视频到未见过的模型的可移植性（一种常见且实用的现实世界场景）仍然有待探索。在本文中，我们率先研究了对抗视频样本在V-MLLM之间的可转移性。我们发现，现有的对抗攻击方法在V-MLLM的黑匣子设置中应用时面临着显着的局限性，我们将其归因于以下缺点：（1）在干扰视频特征方面缺乏一般化，（2）仅关注稀疏关键帧，（3）未能集成多模式信息。为了解决这些限制并加深对黑匣子场景中V-MLLM漏洞的了解，我们引入了图像转视频MLLM（I2 V-MLLM）攻击。在I2 V-MLLM中，我们利用基于图像的多模式大型语言模型（I-MLLM）作为代理模型来制作对抗性视频样本。多模式交互和时空信息被集成，以破坏潜在空间内的视频表示，提高对抗性可转移性。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，我们的方法可以生成对抗性示例，这些示例在多个视频-文本多模式任务上的不同V-MLLM之间表现出很强的可移植性。与对这些模型的白盒攻击相比，我们的黑匣子攻击（使用BLIP-2作为替代模型）实现了有竞争力的性能，对于Zero-Shot VideoQA任务，MSVD-QA的平均攻击成功率（AASB）分别为57.98%和58.26%。



## **47. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

Camera-Ready Version; Accepted at IJCNLP-AACL 2025 Demo

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2412.12478v5) [paper-pdf](https://arxiv.org/pdf/2412.12478v5)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.

摘要: 基于DNN的语言模型在各种NLP任务中表现出色，但仍然极易受到文本对抗攻击。虽然对抗性文本生成对于NLP安全性、可解释性、评估和数据增强至关重要，但相关工作仍然绝大多数以英语为中心，这使得为资源较少的语言构建高质量和可持续的对抗性稳健性基准的问题变得困难且研究不足。首先，由于语言差异和资源有限，资源较低语言的方法定制很复杂。其次，自动攻击容易生成无效或模棱两可的对抗文本。最后但并非最不重要的是，语言模型不断进化，并且可能不受之前生成的部分对抗性文本的影响。为了应对这些挑战，我们引入了HITL-GAT，这是一个基于人机交互生成对抗性文本的通用方法的交互系统。此外，我们还通过藏传文字案例研究展示了HITL-GAT的实用性，采用三种定制的对抗性文本生成方法，并建立了其第一个对抗性鲁棒性基准，为其他资源较少的语言提供了宝贵的参考。



## **48. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

您所看到的并不总是您所得到的：评估GPT对源代码的理解 cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.

摘要: 最近的研究证明了大型语言模型（LLM）在软件工程任务（包括代码生成和理解）中的出色能力。虽然LLM在协助编码方面表现出了巨大的潜力，但LLM很容易受到对抗攻击。在本文中，我们研究了LLM对不可感知攻击的脆弱性。这类攻击在字符级别操纵源代码，这使得更改对人类审查者来说是不可见的，但却有效误导LLM的行为。我们将这些攻击分为四个不同的类别，并分析它们对代码分析和理解任务的影响。这四种不可感知的字符攻击包括编码重新排序、隐形编码字符、代码删除和代码同字形。为了评估最先进的LLM的稳健性，我们使用扰动和干净的代码片段对多个模型进行了系统性评估。引入了两个评估指标，即使用响应的日志概率的模型置信度和响应正确性。结果表明，LLM容易受到不可察觉的编码扰动，不同LLM之间突出显示了不同程度的退化。此外，我们观察到一个一致的扰动幅度和模型性能之间的负相关。这些结果强调了迫切需要强大的LLM能够在难以察觉的对抗条件下操纵行为。



## **49. Exploiting Missing Data Remediation Strategies using Adversarial Missingness Attacks**

使用对抗性缺失攻击利用缺失数据修复策略 cs.LG

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2409.04407v2) [paper-pdf](https://arxiv.org/pdf/2409.04407v2)

**Authors**: Deniz Koyuncu, Alex Gittens, Bülent Yener, Moti Yung

**Abstract**: Adversarial Missingness (AM) attacks aim to manipulate model fitting by carefully engineering a missing data problem to achieve a specific malicious objective. AM attacks are significantly different from prior data poisoning attacks in that no malicious data inserted and no data is maliciously perturbed. Current AM attacks are feasible only under the assumption that the modeler (victim) uses full-information maximum likelihood methods to handle missingness. This work aims to remedy this limitation of AM attacks; in the approach taken here, the adversary achieves their goal by solving a bi-level optimization problem to engineer the adversarial missingness mechanism, where the lower level problem incorporates a differentiable approximation of the targeted missingness remediation technique. As instantiations of this framework, AM attacks are provided for three popular techniques: (i) complete case analysis, (ii) mean imputation, and (iii) regression-based imputation for general empirical risk minimization (ERM) problems. Experiments on real-world data show that AM attacks are successful with modest levels of missingness (less than 20%). Furthermore, we show on the real-world Twins dataset that AM attacks can manipulate the estimated average treatment effect (ATE) as an instance of the general ERM problems: the adversary succeeds in not only reversing the sign, but also in substantially inflating the ATE values from a true value of -1.61% to a manipulated one as high as 10%. These experimental results hold when the ATE is calculated using multiple regression-based estimators with different architectures, even when the adversary is restricted to modifying only a subset of the training data.

摘要: 对抗性缺失（AM）攻击旨在通过精心设计缺失数据问题来操纵模型匹配，以实现特定的恶意目标。AM攻击与之前的数据中毒攻击有显着不同，因为没有插入恶意数据，也没有数据受到恶意干扰。当前的AM攻击只有在建模者（受害者）使用全信息最大似然方法来处理失踪的假设下才可行。这项工作旨在弥补AM攻击的这种局限性;在这里采用的方法中，对手通过解决双层优化问题来设计对抗性缺失机制来实现他们的目标，其中较低层问题结合了目标缺失修复技术的可微逼近。作为该框架的实例，AM攻击为三种流行技术提供：（i）完整案例分析，（ii）均值插补，以及（iii）一般经验风险最小化（ERM）问题的基于回归的插补。对现实世界数据的实验表明，AM攻击是成功的，但丢失率适度（低于20%）。此外，我们在现实世界的Twins数据集上表明，AM攻击可以操纵估计的平均治疗效果（ATE），作为一般ERM问题的一个实例：对手不仅成功扭转了符号，而且还大幅扩大了ATE值从-1.61%的真实值到高达10%的操纵值。当使用具有不同架构的多个基于回归的估计器计算ATE时，即使对手仅限于修改训练数据的子集，这些实验结果也成立。



## **50. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

迪夫保护：使用扩散模型生成对抗示例以保护面部隐私 cs.CV

Code is at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2305.13625v3) [paper-pdf](https://arxiv.org/pdf/2305.13625v3)

**Authors**: Jiang Liu, Chun Pong Lau, Zhongliang Guo, Yuxiang Guo, Zhaoyang Wang, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.

摘要: 日益普及的面部识别（FR）系统引发了人们对个人隐私的严重担忧，尤其是对于在社交媒体上公开分享照片的数十亿用户来说。人们已经做出了多次尝试来保护个人免受未经授权的FR系统利用对抗攻击来生成加密的面部图像的识别。然而，现有方法的视觉质量较差或攻击成功率较低，这限制了它们的实用性。最近，扩散模型在图像生成方面取得了巨大成功。在这项工作中，我们问：扩散模型能否用于生成对抗性示例，以提高视觉质量和攻击性能？我们提出了迪夫保护，它利用扩散自动编码器来在FR系统上生成具有语义意义的扰动。大量实验表明，与最先进的方法相比，迪夫Protect可以生成看起来更自然的加密图像，同时实现显着更高的攻击成功率，例如CelebA-HQ和FFHQ数据集的绝对改进了24.5%和25.1%。



