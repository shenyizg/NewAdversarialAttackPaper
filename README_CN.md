# Latest Adversarial Attack Papers
**update at 2025-07-18 10:43:36**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Automating Steering for Safe Multimodal Large Language Models**

安全多模式大型语言模型的自动转向 cs.CL

Working in progress. 22 pages (8+ for main); 25 figures; 1 table

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13255v1) [paper-pdf](http://arxiv.org/pdf/2507.13255v1)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.

摘要: 多模式大型语言模型（MLLM）的最新进展释放了强大的跨模式推理能力，但也提出了新的安全问题，特别是在面对对抗性多模式输入时。为了提高MLLM在推理过程中的安全性，我们引入了模块化和自适应的推理时干预技术AutoSteer，无需对底层模型进行任何微调。AutoSteer包含三个核心组件：（1）新型安全意识评分（SAS），自动识别模型内部层之间最安全相关的区别;（2）自适应安全探测器，经过训练以估计中间表示有毒输出的可能性;（3）轻量级拒绝头，当检测到安全风险时，它会选择性地干预以调节发电。LLaVa-OG和Chameleon在各种安全关键基准上的实验表明，AutoSteer显着降低了文本、视觉和跨模式威胁的攻击成功率（ASB），同时保持了一般能力。这些发现将AutoSteer定位为一个实用、可解释且有效的框架，用于更安全地部署多模式人工智能系统。



## **2. SHIELD: A Secure and Highly Enhanced Integrated Learning for Robust Deepfake Detection against Adversarial Attacks**

SHIELD：一种安全且高度增强的集成学习，用于针对对抗性攻击的稳健Deepfake检测 cs.SD

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13170v1) [paper-pdf](http://arxiv.org/pdf/2507.13170v1)

**Authors**: Kutub Uddin, Awais Khan, Muhammad Umar Farooq, Khalid Malik

**Abstract**: Audio plays a crucial role in applications like speaker verification, voice-enabled smart devices, and audio conferencing. However, audio manipulations, such as deepfakes, pose significant risks by enabling the spread of misinformation. Our empirical analysis reveals that existing methods for detecting deepfake audio are often vulnerable to anti-forensic (AF) attacks, particularly those attacked using generative adversarial networks. In this article, we propose a novel collaborative learning method called SHIELD to defend against generative AF attacks. To expose AF signatures, we integrate an auxiliary generative model, called the defense (DF) generative model, which facilitates collaborative learning by combining input and output. Furthermore, we design a triplet model to capture correlations for real and AF attacked audios with real-generated and attacked-generated audios using auxiliary generative models. The proposed SHIELD strengthens the defense against generative AF attacks and achieves robust performance across various generative models. The proposed AF significantly reduces the average detection accuracy from 95.49% to 59.77% for ASVspoof2019, from 99.44% to 38.45% for In-the-Wild, and from 98.41% to 51.18% for HalfTruth for three different generative models. The proposed SHIELD mechanism is robust against AF attacks and achieves an average accuracy of 98.13%, 98.58%, and 99.57% in match, and 98.78%, 98.62%, and 98.85% in mismatch settings for the ASVspoof2019, In-the-Wild, and HalfTruth datasets, respectively.

摘要: 音频在扬声器验证、支持语音的智能设备和音频会议等应用中发挥着至关重要的作用。然而，深度造假等音频操纵通过导致错误信息的传播而构成重大风险。我们的实证分析表明，检测Deepfake音频的现有方法通常容易受到反取证（AF）攻击，特别是那些使用生成式对抗网络攻击的攻击。在本文中，我们提出了一种名为SHIELD的新型协作学习方法来防御生成性AF攻击。为了暴露AF签名，我们集成了一个辅助生成模型，称为防御（DF）生成模型，它通过结合输入和输出来促进协作学习。此外，我们设计了一个三元组模型来捕获真实和AF攻击音频与真实生成和攻击生成音频的相关性，使用辅助生成模型。所提出的SHIELD增强了对生成AF攻击的防御，并在各种生成模型中实现了鲁棒性能。对于三种不同的生成模型，所提出的AF将ASVspoof 2019的平均检测准确率从95.49%降至59.77%，将In-the-Wild的平均检测准确率从99.44%降至38.45%，将HalfTruth的平均检测准确率从98.41%降至51.18%。所提出的SHIELD机制对AF攻击具有鲁棒性，并且在ASVspoof 2019，In-the-Wild和HalfTruth数据集的匹配设置中分别实现了98.13%，98.58%和99.57%的平均准确率，以及98.78%，98.62%和98.85%的不匹配设置。



## **3. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

基于属性感知文本反转的身份重识别的可转移对抗攻击 cs.CV

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2502.19697v3) [paper-pdf](http://arxiv.org/pdf/2502.19697v3)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

摘要: 人员重新识别（re-id）模型在安全监控系统中至关重要，需要可转移的对抗攻击来探索其漏洞。近年来，基于视觉语言模型（VLM）的攻击通过攻击VLM的广义图像和文本特征，显示出更好的可移植性，但由于积分表示中过分强调区分性语义，它们缺乏全面的特征破坏。在本文中，我们介绍了属性感知提示攻击（AP-Attack），这是一种新颖的方法，它利用VLM的图像-文本对齐能力，通过破坏特定于属性的文本嵌入来显式破坏行人图像的细粒度语义特征。为了获得各个属性的个性化文本描述，文本倒置网络被设计为将行人图像映射到代表语义嵌入的伪令牌，并通过图像和显式描述行人属性的预定义提示模板以对比学习方式进行训练。反向良性和对抗性细粒度文本语义有助于攻击者有效地进行彻底破坏，增强对抗性示例的可移植性。大量实验表明，AP-Attack实现了最先进的可移植性，在跨模型和数据集攻击场景中的平均丢弃率显着优于之前的方法22.9%。



## **4. Adversarial attacks to image classification systems using evolutionary algorithms**

使用进化算法对图像分类系统的对抗攻击 cs.NE

Genetic and Evolutionary Computation Conference (GECCO '25), July  14--18, 2025, Malaga, Spain

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13136v1) [paper-pdf](http://arxiv.org/pdf/2507.13136v1)

**Authors**: Sergio Nesmachnow, Jamal Toutouh

**Abstract**: Image classification currently faces significant security challenges due to adversarial attacks, which consist of intentional alterations designed to deceive classification models based on artificial intelligence. This article explores an approach to generate adversarial attacks against image classifiers using a combination of evolutionary algorithms and generative adversarial networks. The proposed approach explores the latent space of a generative adversarial network with an evolutionary algorithm to find vectors representing adversarial attacks. The approach was evaluated in two case studies corresponding to the classification of handwritten digits and object images. The results showed success rates of up to 35% for handwritten digits, and up to 75% for object images, improving over other search methods and reported results in related works. The applied method proved to be effective in handling data diversity on the target datasets, even in problem instances that presented additional challenges due to the complexity and richness of information.

摘要: 由于对抗性攻击，图像分类目前面临着重大的安全挑战，对抗性攻击包括旨在欺骗基于人工智能的分类模型的故意更改。本文探讨了一种使用进化算法和生成式对抗网络的组合来生成针对图像分类器的对抗攻击的方法。所提出的方法通过进化算法探索生成式对抗网络的潜在空间，以找到代表对抗攻击的载体。在两个对应于手写数字和对象图像分类的案例研究中对该方法进行了评估。结果显示，手写数字的成功率高达35%，对象图像的成功率高达75%，比其他搜索方法和相关作品中的报告结果有所提高。事实证明，所应用的方法可以有效地处理目标数据集的数据多样性，即使在由于信息的复杂性和丰富性而带来额外挑战的问题实例中也是如此。



## **5. Detecting LLM-generated Code with Subtle Modification by Adversarial Training**

通过对抗训练进行细微修改来检测LLM生成的代码 cs.SE

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13123v1) [paper-pdf](http://arxiv.org/pdf/2507.13123v1)

**Authors**: Xin Yin, Xinrui Li, Chao Ni, Xiaodan Xu, Xiaohu Yang

**Abstract**: With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.

摘要: 随着大型语言模型（LLM）的快速发展，其强大的代码生成能力已被广泛应用于代码完成和自动化开发等任务中，展现了提高编码效率的价值。然而，LLM生成的代码的广泛使用也带来了一些新的挑战。一方面，代码出处监管、版权纠纷和代码质量等问题日益受到关注。如何有效检测LLM生成的代码并确保其合规和负责任地使用已成为一个关键而紧迫的问题。另一方面，在实际应用中，LLM生成的代码经常会受到手动修改，例如变量重命名或结构调整。尽管最近的一些研究提出了基于训练和零射击的方法来检测LLM生成的代码，但这些方法在面对修改后的LLM生成的代码时表现出足够的鲁棒性，并且缺乏有效的解决方案。为了解决LLM生成的代码可能会进行微小修改的现实场景，我们提出了CodeGPTSensor+，这是CodeGPTSensor的增强版本，它采用对抗训练来提高针对输入扰动的鲁棒性。CodeGPTSensor+集成了对抗性样本生成模块多目标标识符和结构转换（MIST），该模块系统地生成高质量和代表性的对抗性样本。该模块有效增强了模型对各种对抗攻击的抵抗力。HMCorp数据集的实验结果表明，CodeGPTSensor+显着提高了对抗测试集的检测准确性，同时保持了原始测试集的高准确性，与CodeGPTSensor相比表现出更强的鲁棒性。



## **6. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

插槽：通过图强化学习进行源驱动APT检测 cs.CR

This paper has been accepted to the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2410.17910v4) [paper-pdf](http://arxiv.org/pdf/2410.17910v4)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.

摘要: 高级持续性威胁（APT）代表复杂的网络攻击，其特征是能够在受害者系统中长时间不被发现，旨在泄露敏感数据或扰乱运营。现有的检测方法常常难以有效识别这些复杂的威胁、构建防御促进的攻击链或抵抗对抗性攻击。为了克服这些挑战，我们提出了Slot，这是一种基于出处图和图强化学习的高级APT检测方法。Slot擅长通过出处图挖掘发现系统行为之间的多层隐藏关系，例如因果关系、上下文关系和间接联系。通过开创图强化学习的集成，Slot动态适应新的用户活动和不断发展的攻击策略，增强其对对抗性攻击的弹性。此外，Slot还根据检测到的攻击，通过集群算法自动构建攻击链，提供攻击路径的精确识别，促进防御策略的制定。对现实世界数据集的评估证明了Slot在APT检测方面出色的准确性、效率、适应性和稳健性，大多数指标都超越了最先进的方法。此外，为评估Slot支持APT防御的有效性而进行的案例研究进一步确立了其作为网络安全保护实用且可靠的工具的地位。



## **7. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

用于增强对抗可移植性的语义结构感知生成攻击 cs.CV

Preprint

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2506.18248v3) [paper-pdf](http://arxiv.org/pdf/2506.18248v3)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).

摘要: 生成性对抗攻击在白盒代理模型上训练扰动生成器，然后将精心设计的扰动应用于不可见的黑匣子受害者模型。与迭代攻击相比，这些方法提供了卓越的推理时效率、可扩展性和可移植性;然而，到目前为止，现有的研究尚未充分利用生成模型的表示能力来保存和利用语义信息。具体来说，生成器的中间激活编码了丰富的语义特征--对象边界和粗糙形状--这些特征仍然没有得到充分利用，从而限制了扰动与对象突出区域的对齐，而这些区域对于对抗性的可转移性至关重要。为了解决这个问题，我们引入了一个基于Mean Teacher的语义结构感知攻击框架，该框架充当时间平滑的特征参考。通过这个平滑的引用，我们通过特征提炼进一步指导学生的早期层激活和语义丰富的教师的早期层激活之间的语义一致性。通过基于经验发现将扰动合成锚定到生成器内语义突出的早期中间块，我们的方法引导对区域进行渐进对抗扰动，从而大大增强对抗转移性。我们对不同的模型、领域和任务进行了广泛的实验，以展示相对于最先进的生成性攻击的一致改进，并使用传统指标和我们新提出的意外纠正率（OCR）进行了全面评估。



## **8. A Bayesian Incentive Mechanism for Poison-Resilient Federated Learning**

抗毒联邦学习的Bayesian激励机制 cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12439v1) [paper-pdf](http://arxiv.org/pdf/2507.12439v1)

**Authors**: Daniel Commey, Rebecca A. Sarpong, Griffith S. Klogo, Winful Bagyl-Bac, Garth V. Crosby

**Abstract**: Federated learning (FL) enables collaborative model training across decentralized clients while preserving data privacy. However, its open-participation nature exposes it to data-poisoning attacks, in which malicious actors submit corrupted model updates to degrade the global model. Existing defenses are often reactive, relying on statistical aggregation rules that can be computationally expensive and that typically assume an honest majority. This paper introduces a proactive, economic defense: a lightweight Bayesian incentive mechanism that makes malicious behavior economically irrational. Each training round is modeled as a Bayesian game of incomplete information in which the server, acting as the principal, uses a small, private validation dataset to verify update quality before issuing payments. The design satisfies Individual Rationality (IR) for benevolent clients, ensuring their participation is profitable, and Incentive Compatibility (IC), making poisoning an economically dominated strategy. Extensive experiments on non-IID partitions of MNIST and FashionMNIST demonstrate robustness: with 50% label-flipping adversaries on MNIST, the mechanism maintains 96.7% accuracy, only 0.3 percentage points lower than in a scenario with 30% label-flipping adversaries. This outcome is 51.7 percentage points better than standard FedAvg, which collapses under the same 50% attack. The mechanism is computationally light, budget-bounded, and readily integrates into existing FL frameworks, offering a practical route to economically robust and sustainable FL ecosystems.

摘要: 联合学习（FL）支持跨去中心化客户端的协作模型训练，同时保护数据隐私。然而，其开放参与性质使其面临数据中毒攻击，其中恶意行为者提交损坏的模型更新以降级全局模型。现有的防御通常是反应性的，依赖于统计聚合规则，这些规则在计算上可能很昂贵，并且通常假设诚实多数。本文引入了一种积极主动的经济防御：一种轻量级的Bayesian激励机制，使恶意行为在经济上变得不合理。每个训练轮都被建模为不完整信息的Bayesian博弈，其中服务器作为委托人，使用一个小型的私人验证数据集来在发放付款之前验证更新质量。该设计满足仁慈客户的个人理性（IR），确保他们的参与有利可图，并满足激励兼容性（IC），使投毒成为经济主导策略。对MNIST和FashionMNIST的非IID分区的广泛实验证明了鲁棒性：MNIST上有50%的标签翻转对手，该机制保持96.7%的准确率，仅比30%的标签翻转对手的情况低0.3个百分点。这一结果比标准FedAverage好51.7个百分点，后者在同样的50%攻击下崩溃。该机制计算量轻、预算有限，并且易于集成到现有的FL框架中，为实现经济稳健和可持续的FL生态系统提供了一条实用的途径。



## **9. Trustworthy Tree-based Machine Learning by $MoS_2$ Flash-based Analog CAM with Inherent Soft Boundaries**

值得信赖的基于树的机器学习，由$MoS_2 $基于Flash的模拟CAM实现，具有固有软边界 cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12384v1) [paper-pdf](http://arxiv.org/pdf/2507.12384v1)

**Authors**: Bo Wen, Guoyun Gao, Zhicheng Xu, Ruibin Mao, Xiaojuan Qi, X. Sharon Hu, Xunzhao Yin, Can Li

**Abstract**: The rapid advancement of artificial intelligence has raised concerns regarding its trustworthiness, especially in terms of interpretability and robustness. Tree-based models like Random Forest and XGBoost excel in interpretability and accuracy for tabular data, but scaling them remains computationally expensive due to poor data locality and high data dependence. Previous efforts to accelerate these models with analog content addressable memory (CAM) have struggled, due to the fact that the difficult-to-implement sharp decision boundaries are highly susceptible to device variations, which leads to poor hardware performance and vulnerability to adversarial attacks. This work presents a novel hardware-software co-design approach using $MoS_2$ Flash-based analog CAM with inherent soft boundaries, enabling efficient inference with soft tree-based models. Our soft tree model inference experiments on $MoS_2$ analog CAM arrays show this method achieves exceptional robustness against device variation and adversarial attacks while achieving state-of-the-art accuracy. Specifically, our fabricated analog CAM arrays achieve $96\%$ accuracy on Wisconsin Diagnostic Breast Cancer (WDBC) database, while maintaining decision explainability. Our experimentally calibrated model validated only a $0.6\%$ accuracy drop on the MNIST dataset under $10\%$ device threshold variation, compared to a $45.3\%$ drop for traditional decision trees. This work paves the way for specialized hardware that enhances AI's trustworthiness and efficiency.

摘要: 人工智能的快速发展引发了人们对其可信度的担忧，特别是在可解释性和稳健性方面。Random Forest和XGboost等基于树的模型在表格数据的可解释性和准确性方面表现出色，但由于数据局部性差和数据依赖性高，扩展它们的计算成本仍然很高。之前使用模拟内容可编址存储器（CAM）加速这些模型的努力一直很困难，因为难以实现的尖锐决策边界极易受到设备变化的影响，从而导致硬件性能较差并容易受到对抗性攻击。这项工作提出了一种新颖的硬件-软件协同设计方法，使用具有固有软边界的基于Flash的模拟CAM，实现了基于软树的模型的高效推理。我们在$MoS_2 $模拟CAM阵列上进行的软树模型推理实验表明，该方法在实现最先进的准确性的同时，对设备变化和对抗性攻击具有出色的鲁棒性。具体来说，我们制造的模拟CAM阵列在威斯康星州诊断乳腺癌（WDBC）数据库中实现了96%美元的准确性，同时保持决策的可解释性。我们的实验校准模型验证了在10美元的设备阈值变化下，MNIST数据集的准确性仅下降了0.6%，而传统决策树的准确性下降了45.3%。这项工作为提高人工智能可信度和效率的专用硬件铺平了道路。



## **10. Thought Purity: Defense Paradigm For Chain-of-Thought Attack**

思想纯洁性：思想链攻击的防御范式 cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12314v1) [paper-pdf](http://arxiv.org/pdf/2507.12314v1)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.

摘要: 虽然强化学习训练的大型推理模型（LRM，例如，Deepseek-R1）在不断发展的大型语言模型（LLM）领域展示了先进的推理能力，但它们对安全威胁的敏感性仍然是一个关键的漏洞。这种弱点在思想链（CoT）生成过程中尤为明显，其中后门提示攻击等对抗性方法可以系统地颠覆模型的核心推理机制。新兴的思想链攻击（CoTA）通过利用即时可控性来揭示了这一漏洞，同时通过低成本干预降低CoT的安全性和任务性能。为了解决这种复杂的安全性能漏洞，我们提出了Thought Purity（TP）：一种防御范式，可以系统性地增强对恶意内容的抵抗力，同时保持运营效率。我们的解决方案通过三个协同组件实现这一目标：（1）安全优化的数据处理管道（2）强化学习增强的规则约束（3）自适应监控指标。我们的方法在强化学习一致的推理系统中建立了第一个针对CoTA漏洞的全面防御机制，显着推进了下一代人工智能架构的安全功能平衡。



## **11. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

用于机械化安全编译证明的调用返回树的Nanopass反向翻译 cs.PL

ITP'25 final version

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2503.19609v3) [paper-pdf](http://arxiv.org/pdf/2503.19609v3)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.

摘要: 研究人员的目标是构建安全的编译链，确保如果源上下文不存在针对源程序的攻击，那么对抗性目标上下文也不存在针对已编译的程序的攻击。然而，证明这些编译链是安全的是具有挑战性的，并且涉及一个不平凡的反向翻译步骤：对于目标上下文针对已编译的程序发起的任何攻击，必须展示针对源程序发起的相同攻击的源上下文。我们描述了一种新颖的反向翻译技术，它可以产生更简单的证明，并且可以更容易地在证明助手中机械化。给定一组有限的有限跟踪前置，捕获攻击期间目标上下文和已编译的程序之间记录的交互，我们构建一个调用返回树，将其反向翻译为源上下文，从而产生相同的跟踪前置。我们使用生成的源上下文中的状态来记录调用返回树中的当前位置。反向翻译是分几个小步骤完成的，每个步骤都会向树中添加新信息，描述位置应该如何根据上下文如何重新获得控制而更改。为了证明这种反向翻译是正确的，我们为每个中间调用返回树语言赋予语义，使用幽灵状态来存储信息并显式地强制执行执行不变量。我们证明了几个小型的正向模拟，基本上将反向翻译视为经过验证的纳米ass编译器。由于这种模块化结构，我们能够在Rocq证明器中机械化这种复杂的反向翻译及其正确性证明，而无需太多努力。



## **12. Non-Adaptive Adversarial Face Generation**

非适应性对抗人脸生成 cs.CV

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12107v1) [paper-pdf](http://arxiv.org/pdf/2507.12107v1)

**Authors**: Sunpill Kim, Seunghun Paik, Chanwoo Hwang, Minsu Kim, Jae Hong Seo

**Abstract**: Adversarial attacks on face recognition systems (FRSs) pose serious security and privacy threats, especially when these systems are used for identity verification. In this paper, we propose a novel method for generating adversarial faces-synthetic facial images that are visually distinct yet recognized as a target identity by the FRS. Unlike iterative optimization-based approaches (e.g., gradient descent or other iterative solvers), our method leverages the structural characteristics of the FRS feature space. We figure out that individuals sharing the same attribute (e.g., gender or race) form an attributed subsphere. By utilizing such subspheres, our method achieves both non-adaptiveness and a remarkably small number of queries. This eliminates the need for relying on transferability and open-source surrogate models, which have been a typical strategy when repeated adaptive queries to commercial FRSs are impossible. Despite requiring only a single non-adaptive query consisting of 100 face images, our method achieves a high success rate of over 93% against AWS's CompareFaces API at its default threshold. Furthermore, unlike many existing attacks that perturb a given image, our method can deliberately produce adversarial faces that impersonate the target identity while exhibiting high-level attributes chosen by the adversary.

摘要: 对人脸识别系统（FRS）的对抗攻击构成严重的安全和隐私威胁，尤其是当这些系统用于身份验证时。在本文中，我们提出了一种生成敌对面部的新方法--视觉上不同但被FSG识别为目标身份的合成面部图像。与基于迭代优化的方法不同（例如，梯度下降或其他迭代求解器），我们的方法利用了FSG特征空间的结构特征。我们发现共享相同属性的个人（例如，性别或种族）形成归属子领域。通过利用此类子领域，我们的方法既实现了非适应性，又实现了数量非常少的查询。这消除了依赖可移植性和开源代理模型的需要，这是当不可能对商业FIS进行重复自适应查询时的典型策略。尽管仅需要由100个面部图像组成的单个非自适应查询，但我们的方法在默认阈值下针对AWS的CompareFaces API实现了超过93%的高成功率。此外，与许多干扰给定图像的现有攻击不同，我们的方法可以故意产生模仿目标身份的对抗面孔，同时展示对手选择的高级属性。



## **13. Distributed Resilient State Estimation and Control with Strategically Implemented Security Measures**

具有战略性实施的安全措施的分布式弹性状态估计和控制 eess.SY

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12052v1) [paper-pdf](http://arxiv.org/pdf/2507.12052v1)

**Authors**: Takumi Shinohara, Karl H. Johansson, Henrik Sandberg

**Abstract**: This paper addresses the problem of distributed resilient state estimation and control for linear time-invariant systems in the presence of malicious false data injection sensor attacks and bounded noise. We consider a system operator (defender) capable of deploying cybersecurity measures to counteract the sensor compromises. Although such measures enhance resilience against adversarial attacks, they may incur substantial costs; hence, it is crucial to select countermeasures to balance resilience gains and cost efficiency strategically. We first demonstrate that the system's resilience against attacks is maximized through the appropriate implementation of security measures, implying that no attacker can execute undetectable sensor attacks. Building on this analysis, we propose an algorithm that identifies the optimal security measure. While determining this measure is NP-hard in general, we also derive sufficient conditions under which efficient computation is feasible. Furthermore, we develop a distributed resilient state estimation and control scheme informed by the optimal security measure and establish conditions that guarantee bounded estimation and control errors. Finally, we validate the efficacy of our approach via numerical simulations of a vehicle platooning scenario.

摘要: 本文研究了存在恶意虚假数据注入传感器攻击和有界噪音的线性时不变系统的分布式弹性状态估计和控制问题。我们认为系统运营商（防御者）能够部署网络安全措施来抵消传感器的损害。尽管此类措施增强了抵御敌对攻击的弹性，但可能会产生巨大的成本;因此，从战略上选择应对措施来平衡弹性收益和成本效率至关重要。我们首先证明，通过适当实施安全措施，系统对攻击的弹性得到了最大化，这意味着攻击者无法执行无法检测到的传感器攻击。在此分析的基础上，我们提出了一种识别最佳安全措施的算法。虽然确定这个测量通常是NP困难的，但我们也推导出有效计算可行的充分条件。此外，我们开发了一种由最佳安全措施通知的分布式弹性状态估计和控制方案，并建立保证有界估计和控制误差的条件。最后，我们通过车辆排队场景的数字模拟验证了我们方法的有效性。



## **14. Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation**

观看、聆听、理解、误导：对短视频进行三模式对抗攻击以进行内容适当性评估 cs.CV

Accepted as long paper, SVU Workshop at ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11968v1) [paper-pdf](http://arxiv.org/pdf/2507.11968v1)

**Authors**: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs.

摘要: 多模式大型语言模型（MLLM）越来越多地用于内容审核，但其在简短视频上下文中的稳健性仍然没有得到充分的探索。当前的安全评估通常依赖于单模式攻击，未能解决组合攻击漏洞。在本文中，我们介绍了一个用于评估MLLM三峰安全性的全面框架。首先，我们展示了短视频多模式对抗（SVMA）数据集，其中包括具有人类引导的合成对抗攻击的各种短视频。其次，我们提出了ChimeraBreak，这是一种新型的三模式攻击策略，同时挑战视觉、听觉和语义推理途径。对最先进的MLLM进行的广泛实验揭示了具有高攻击成功率（ASB）的重大漏洞。我们的研究结果揭示了不同的失败模式，表明模型对良性或违反政策的内容进行错误分类的偏见。我们使用LLM作为法官来评估结果，证明了攻击推理的有效性。我们的数据集和调查结果为开发更强大、更安全的MLLM提供了重要见解。



## **15. When and Where do Data Poisons Attack Textual Inversion?**

数据毒药何时何地攻击文本倒置？ cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.10578v2) [paper-pdf](http://arxiv.org/pdf/2507.10578v2)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10

摘要: 中毒攻击对扩散模型（DM）的鲁棒性构成了重大挑战。本文系统地分析了中毒攻击文本倒置（TI）的时间和地点，文本倒置（TI）是一种广泛使用的DM个性化技术。我们首先介绍语义敏感度地图，这是一种用于可视化中毒对文本嵌入影响的新颖方法。其次，我们识别并通过实验验证DM在跨时间步上表现出非均匀的学习行为，重点关注低噪音样本。中毒攻击继承了这种偏见，并主要在较低的时间步注入对抗信号。最后，我们观察到对抗信号分散了学习对训练数据中相关概念区域的注意力，从而破坏了TI过程。基于这些见解，我们提出了安全区训练（SZT），这是一种由3个关键组件组成的新型防御机制：（1）JPEG压缩以削弱高频毒物信号，（2）TI训练期间限制高时步，以避免较低时步的对抗信号，（3）损失掩蔽以将学习限制在相关区域。多种中毒方法的广泛实验表明，SZT极大地增强了TI针对所有中毒攻击的稳健性，提高了生成质量，超出了之前发布的防御措施。代码：www.github.com/JStyborski/Diff_Lab数据：www.github.com/JStyborski/NC10



## **16. LLMs Encode Harmfulness and Refusal Separately**

法学硕士分别对有害和拒绝进行编码 cs.CL

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11878v1) [paper-pdf](http://arxiv.org/pdf/2507.11878v1)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety

摘要: LLM接受过拒绝有害指令的培训，但他们真正了解除了拒绝之外的危害吗？先前的工作表明，LLM的拒绝行为可以通过一维子空间来调节，即拒绝方向。在这项工作中，我们确定了一个新的维度来分析LLM中的安全机制，即危害性，它在内部被编码为与拒绝分开的概念。存在一个与拒绝方向不同的危害方向。作为因果证据，沿着有害方向引导可能会导致LLM将无害的指令解释为有害的，但沿着拒绝方向引导往往会直接引发拒绝反应，而不会扭转模型对有害性的判断。此外，使用我们确定的危害性概念，我们发现某些越狱方法通过减少拒绝信号来发挥作用，而不会扭转模型内部的危害性信念。我们还发现，对抗性地微调模型以接受有害的指令对模型的内部有害信念的影响最小。这些见解导致了一个实际的安全应用：该模型的潜在危害表示可以作为一个内在的保障（潜在的警卫），用于检测不安全的输入和减少过度拒绝，这是强大的微调攻击。例如，我们的Latent Guard在不同的越狱方法中实现了与Llama Guard 3 8B相当或更好的性能，Llama Guard 3 8B是一种专用的微调保护模型。我们的研究结果表明，LLM对危害性的内部理解比他们拒绝不同输入指令的决定更强大，为研究AI安全性提供了一个新的视角



## **17. Some remarks on gradient dominance and LQR policy optimization**

关于梯度主导和LQR政策优化的一些看法 cs.LG

This is a short paper summarizing the first part of the slides  presented at my keynote at the 2025 L4DC (Learning for Dynamics & Control  Conference) in Ann Arbor, Michigan, 05 June 2025. A partial bibliography has  been added

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.10452v2) [paper-pdf](http://arxiv.org/pdf/2507.10452v2)

**Authors**: Eduardo D. Sontag

**Abstract**: Solutions of optimization problems, including policy optimization in reinforcement learning, typically rely upon some variant of gradient descent. There has been much recent work in the machine learning, control, and optimization communities applying the Polyak-{\L}ojasiewicz Inequality (PLI) to such problems in order to establish an exponential rate of convergence (a.k.a. ``linear convergence'' in the local-iteration language of numerical analysis) of loss functions to their minima under the gradient flow. Often, as is the case of policy iteration for the continuous-time LQR problem, this rate vanishes for large initial conditions, resulting in a mixed globally linear / locally exponential behavior. This is in sharp contrast with the discrete-time LQR problem, where there is global exponential convergence. That gap between CT and DT behaviors motivates the search for various generalized PLI-like conditions, and this talk will address that topic. Moreover, these generalizations are key to understanding the transient and asymptotic effects of errors in the estimation of the gradient, errors which might arise from adversarial attacks, wrong evaluation by an oracle, early stopping of a simulation, inaccurate and very approximate digital twins, stochastic computations (algorithm ``reproducibility''), or learning by sampling from limited data. We describe an ``input to state stability'' (ISS) analysis of this issue. The second part discusses convergence and PLI-like properties of ``linear feedforward neural networks'' in feedback control. Much of the work described here was done in collaboration with Arthur Castello B. de Oliveira, Leilei Cui, Zhong-Ping Jiang, and Milad Siami.

摘要: 优化问题的解决方案，包括强化学习中的策略优化，通常依赖于梯度下降的某种变体。机器学习、控制和优化领域最近有很多工作将Polyak-{\L}ojasiewicz不等式（PLI）应用于此类问题，以建立指数收敛率（又名数值分析的局部迭代语言中的“线性收敛”）损失函数在梯度流下达到最小值。通常情况下，作为连续时间LQR问题的策略迭代的情况下，该速率为大的初始条件消失，导致混合全局线性/局部指数行为。这与离散时间LQR问题形成鲜明对比，其中存在全局指数收敛。CT和DT行为之间的差距激发了对各种广义的类神经元疾病的研究，本次演讲将讨论这个话题。此外，这些概括对于理解梯度估计中的误差的瞬态和渐近效应是关键的，这些误差可能来自对抗性攻击，预言机的错误评估，模拟的早期停止，不准确和非常近似的数字孪生，随机计算（算法“可再现性”），或通过从有限的数据中采样进行学习。我们描述了对这个问题的“国家稳定的投入”（ISS）分析。第二部分讨论反馈控制中“线性前向神经网络”的收敛性和类锯齿性质。这里描述的大部分工作是与Arthur Castello B合作完成的。de Oliveira、Leilei Cui、Jiang Ping和Milad Siami。



## **18. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

使用特殊红旗令牌进行LLM危害检测的生成方法 cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.

摘要: 大型语言模型（LLM）的大多数安全训练方法都基于微调，迫使模型在面临有害请求时从不安全的答案转向拒绝。不幸的是，这些急剧的分布变化通常会损害模型的能力。为了避免这种情况，我们建议使用一个我们称为红旗令牌（）的特殊令牌来扩展模型的词汇表<rf>，并建议训练模型，以便在生成或即将生成有害内容时随时将此令牌插入到其响应中。我们的方法提供了几个优点：它使模型能够明确地学习危害性的概念，同时对生成的分布产生轻微影响，从而保持模型的效用。它还评估每个生成的答案，并提供与对抗训练一样好的鲁棒性，而无需在训练期间运行攻击。此外，通过将我们的安全调优封装在LoRA模块中，我们提供了针对微调API攻击的额外防御。



## **19. Robustifying 3D Perception via Least-Squares Graphs for Multi-Agent Object Tracking**

基于最小二乘图的3D感知鲁棒化多智能体目标跟踪 cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.04762v2) [paper-pdf](http://arxiv.org/pdf/2507.04762v2)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.

摘要: EdgeAI系统（例如自动驾驶汽车）的关键感知能力需要能够随着时间的推移准确识别和定位场景中的多个对象，从而减轻其影响，从而能够抵御对抗威胁。单代理跟踪提供了对抗攻击的弹性，但缺乏情景感知，这凸显了多代理合作以增强上下文理解和稳健性的必要性。本文提出了一种针对对抗性噪音的新型缓解框架，通过基于多智能体对抗性边界盒上的最小平方图跟踪对象。具体来说，我们使用最小平方图形工具来利用通过差坐标和锚点的完全连接图形上的重叠边界框来减少每个检测的重心的诱导位置误差。因此，多车辆检测被融合和细化，以减轻对抗影响，并在两个阶段与现有轨道相关联，执行跟踪，以进一步抑制对抗威胁。对现实世界V2 V4 Real数据集的广泛评估研究表明，在具有挑战性的对抗条件下，所提出的方法显着比最先进的单代理和多代理跟踪框架高出23.3%，作为一种弹性方法运行，无需依赖额外的防御机制。



## **20. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

（图）神经网络对抗数据中毒和后门攻击的可证明鲁棒性 cs.LG

Published in TMLR. Best Paper Award at the AdvML-Frontiers @ NeurIPS  2024 workshop. Code available at https://github.com/saper0/qpcert

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2407.10867v3) [paper-pdf](http://arxiv.org/pdf/2407.10867v3)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data. This vulnerability has led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning attacks, including backdoors, targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.

摘要: 机器学习模型的泛化可能会受到数据中毒的严重影响，其中对抗性的变化被应用于训练数据。这种脆弱性导致了对认证的兴趣（即，证明）达到一定量级的这种变化不影响试验预测。我们第一次证明了图神经网络（GNN）可以抵御针对给定图的节点特征的中毒攻击，包括后门。我们的证书是白盒的，基于$（i）$神经切核，它描述了足够宽的网络的训练动态;和$（ii）$二层优化问题的新颖重新表述，将中毒描述为混合整线性规划。因此，我们利用我们的框架来提供有关图结构及其连接性对基于卷积和基于PageRank的GNN的最坏情况稳健性行为的作用的基本见解。我们注意到，我们的框架更加通用，并且构成了第一种为NN推导白盒中毒证书的方法，这可能具有超越图形相关任务的独立兴趣。



## **21. Real-Time Bayesian Detection of Drift-Evasive GNSS Spoofing in Reinforcement Learning Based UAV Deconfliction**

基于强化学习的无人机去冲突中漂移规避GPS欺骗的实时Bayesian检测 cs.LG

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11173v1) [paper-pdf](http://arxiv.org/pdf/2507.11173v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: Autonomous unmanned aerial vehicles (UAVs) rely on global navigation satellite system (GNSS) pseudorange measurements for accurate real-time localization and navigation. However, this dependence exposes them to sophisticated spoofing threats, where adversaries manipulate pseudoranges to deceive UAV receivers. Among these, drift-evasive spoofing attacks subtly perturb measurements, gradually diverting the UAVs trajectory without triggering conventional signal-level anti-spoofing mechanisms. Traditional distributional shift detection techniques often require accumulating a threshold number of samples, causing delays that impede rapid detection and timely response. Consequently, robust temporal-scale detection methods are essential to identify attack onset and enable contingency planning with alternative sensing modalities, improving resilience against stealthy adversarial manipulations. This study explores a Bayesian online change point detection (BOCPD) approach that monitors temporal shifts in value estimates from a reinforcement learning (RL) critic network to detect subtle behavioural deviations in UAV navigation. Experimental results show that this temporal value-based framework outperforms conventional GNSS spoofing detectors, temporal semi-supervised learning frameworks, and the Page-Hinkley test, achieving higher detection accuracy and lower false-positive and false-negative rates for drift-evasive spoofing attacks.

摘要: 自主无人飞行器（UF）依赖全球导航卫星系统（GNSS）伪距测量来实现准确的实时定位和导航。然而，这种依赖性使它们面临复杂的欺骗威胁，对手操纵伪距来欺骗无人机接收器。其中，漂移规避欺骗攻击微妙地扰乱测量，逐渐改变无人机的轨迹，而不会触发传统的信号级反欺骗机制。传统的分布式漂移检测技术通常需要积累阈值数量的样本，从而导致延迟，阻碍快速检测和及时响应。因此，强大的时间尺度检测方法对于识别攻击发作并使用替代传感模式进行应急计划至关重要，从而提高对隐形对抗操纵的弹性。本研究探讨了贝叶斯在线变点检测（BOCPD）方法，该方法可以监控强化学习（RL）评论家网络的值估计值的时间变化，以检测无人机导航中的细微行为偏差。实验结果表明，该框架优于传统的GNSS欺骗检测器、时间半监督学习框架和Page-Hinkley测试，实现了更高的检测精度和更低的漂移规避欺骗攻击误报率和漏报率。



## **22. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

多触发中毒放大了LLM中的后门漏洞 cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.

摘要: 最近的研究表明，大型语言模型（LLM）容易受到数据中毒攻击，其中恶意训练示例嵌入了由特定输入模式触发的隐藏行为。然而，大多数现有的作品假设一个短语，并专注于攻击的有效性，提供有限的理解触发机制和多个触发器如何在模型中相互作用。在本文中，我们提出了一个框架，研究中毒的LLM。我们发现，多个不同的后门触发器可以共存于一个单一的模型中，而不会相互干扰，使对手能够同时嵌入多个触发器。使用具有高嵌入相似性的多个触发器，我们证明即使令牌被长令牌跨度替换或分开，中毒触发器也可以实现稳健的激活。我们的研究结果揭示了LLC中更广泛、更持久的脆弱性表面。为了减轻这种威胁，我们提出了一种事后恢复方法，该方法根据分层权重差异分析选择性地重新训练特定的模型组件。我们的方法通过最少的参数更新有效地消除了触发行为，从而提供了针对多触发中毒的实用有效防御。



## **23. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

面具背后的魔鬼：扩散LLC的紧急安全漏洞 cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.

摘要: 基于扩散的大型语言模型（dLLM）最近成为自回归LLM的强大替代方案，通过并行解码和双向建模提供更快的推理和更强的交互性。然而，尽管在代码生成和文本填充方面表现出色，但我们发现了一个基本的安全问题：现有的对齐机制未能保护dLLM免受上下文感知、屏蔽输入对抗提示的影响，从而暴露了新颖的漏洞。为此，我们提出了DIJA，这是第一个利用dLLM独特安全弱点的系统性研究和越狱攻击框架。具体来说，我们提出的DIJA构建了对抗性交错屏蔽文本提示，利用dLLM的文本生成机制，即双向建模和并行解码。双向建模驱动模型为掩蔽跨度生成上下文一致的输出，即使是有害的，而并行解码限制了模型动态过滤和不安全内容的拒绝采样。这会导致标准对齐机制失败，从而导致在经过优化的DLLM中进行有害的完成，即使在提示中直接暴露了有害行为或不安全的指令。通过全面的实验，我们证明DIJA的性能显着优于现有的越狱方法，暴露了dLLM架构中之前被忽视的威胁表面。值得注意的是，我们的方法在Dream-Direct上实现了高达100%的基于关键词的ASB，超过了最强的先前基线ReNeLLM，在JailbreakBench上基于评估者的ASB中提高了高达78.5%，在StrongRESYS评分中提高了37.7分，同时不需要重写或隐藏越狱提示中的有害内容。我们的研究结果强调了重新思考这类新兴语言模型中的安全一致的迫切需要。代码可在https://github.com/ZichenWen1/DIJA上获取。



## **24. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

针对表格数据设计不可感知的Manifold对抗攻击 cs.LG

32 pages

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10998v1) [paper-pdf](http://arxiv.org/pdf/2507.10998v1)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present fundamental challenges distinct from image or text domains due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions, making them detectable. We propose a latent space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate imperceptible adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We specify In-Distribution Success Rate (IDSR) to measure the proportion of adversarial examples that remain statistically indistinguishable from the input distribution. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches. Our comprehensive analysis includes hyperparameter sensitivity, sparsity control mechanisms, and generative architectural comparisons, revealing that VAE-based attacks depend critically on reconstruction quality but offer superior practical utility when sufficient training data is available. This work highlights the importance of on-manifold perturbations for realistic adversarial attacks on tabular data, offering a robust approach for practical deployment. The source code can be accessed through https://github.com/ZhipengHe/VAE-TabAttack.

摘要: 由于混合类别和数字特征的异类性质，对表格数据的对抗性攻击带来了与图像或文本域不同的根本挑战。与像素扰动保持视觉相似性的图像不同，表格数据缺乏直观的相似性指标，因此很难定义难以察觉的修改。此外，传统的基于梯度的方法优先考虑$\ell_p$-norm约束，通常会产生偏离原始数据分布的对抗性示例，使其可检测。我们提出了一个潜在空间扰动框架，使用混合输入变分自动编码器（VAE）来生成难以察觉的对抗示例。拟议的VAE集成分类嵌入和数值特征到一个统一的潜在流形，使扰动保持统计一致性。我们指定分布内成功率（IDSR）来衡量在统计上与输入分布无法区分的对抗性示例的比例。对六个公开可用的数据集和三个模型架构的评估表明，与传统的输入空间攻击和其他基于图像域方法的VAE方法相比，我们的方法实现了更低的离群值率和更一致的性能。我们的全面分析包括超参数敏感性，稀疏控制机制和生成架构比较，揭示了基于VAE的攻击严重依赖于重建质量，但在有足够的训练数据时提供了卓越的实际效用。这项工作强调了对表格数据进行现实对抗攻击的管汇上扰动的重要性，为实际部署提供了一种稳健的方法。源代码可通过https://github.com/ZhipengHe/VAE-TabAttack访问。



## **25. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。最近的对抗攻击、微调漏洞以及在高风险环境中增加部署LLM可能会放大这些风险。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新颖的方法，它从根本上破坏了LLM中有害行为的潜在表现，提供了可扩展的解决方案来增强（潜在固有的）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **26. A Survey on Speech Deepfake Detection**

语音Deepfake检测综述 cs.SD

38 pages. This paper has been accepted by ACM Computing Surveys

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2404.13914v2) [paper-pdf](http://arxiv.org/pdf/2404.13914v2)

**Authors**: Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang

**Abstract**: The availability of smart devices leads to an exponential increase in multimedia content. However, advancements in deep learning have also enabled the creation of highly sophisticated Deepfake content, including speech Deepfakes, which pose a serious threat by generating realistic voices and spreading misinformation. To combat this, numerous challenges have been organized to advance speech Deepfake detection techniques. In this survey, we systematically analyze more than 200 papers published up to March 2024. We provide a comprehensive review of each component in the detection pipeline, including model architectures, optimization techniques, generalizability, evaluation metrics, performance comparisons, available datasets, and open source availability. For each aspect, we assess recent progress and discuss ongoing challenges. In addition, we explore emerging topics such as partial Deepfake detection, cross-dataset evaluation, and defences against adversarial attacks, while suggesting promising research directions. This survey not only identifies the current state of the art to establish strong baselines for future experiments but also offers clear guidance for researchers aiming to enhance speech Deepfake detection systems.

摘要: 智能设备的可用性导致多媒体内容呈指数级增长。然而，深度学习的进步也使得能够创建高度复杂的Deepfake内容，包括语音Deepfake，这些内容通过生成真实的声音和传播错误信息构成严重威胁。为了解决这个问题，人们组织了许多挑战来推进语音Deepfake检测技术。在这项调查中，我们系统分析了截至2024年3月发表的200多篇论文。我们对检测管道中的每个组件进行全面审查，包括模型架构、优化技术、可概括性、评估指标、性能比较、可用数据集和开源可用性。对于每个方面，我们都会评估最近的进展并讨论当前的挑战。此外，我们还探索了部分Deepfake检测、跨数据集评估和对抗性攻击防御等新兴主题，同时提出了有前途的研究方向。这项调查不仅确定了当前的最新技术水平，为未来的实验建立强有力的基线，而且还为旨在增强语音Deepfake检测系统的研究人员提供了明确的指导。



## **27. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

REAL-IOT：描述实际对抗攻击下GNN入侵检测的鲁棒性 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.

摘要: 基于图形神经网络（GNN）的网络入侵检测系统（NIDS）通常在单个数据集上进行评估，这限制了它们在分布漂移下进行概括的能力。此外，它们的对抗鲁棒性通常是使用缺乏真实性的合成扰动来评估的。这种测量差距导致高估了基于GNN的NIDS弹性。为了解决这些局限性，我们提出了\textBF{REAL-IoT}，这是一个用于在物联网环境中对基于GNN的NIDS进行稳健性评估的综合框架。我们的框架提出了一种方法，该方法从规范数据集创建统一数据集，以评估漂移下的概括性。此外，它还具有从物理物联网测试台收集的新型入侵数据集，该数据集可以捕获现实世界环境下的网络流量和攻击场景。此外，使用REAL-IOT，我们探索使用大型语言模型（LLM）来分析网络数据并通过过滤可疑流来减轻对抗示例的影响。我们使用REAL-IOT进行的评估显示，与标准基准的结果相比，GNN模型的性能有所下降，量化了它们对漂移和现实攻击的敏感性。我们还展示了基于LLM的过滤增强鲁棒性的潜力。这些发现强调了开发弹性物联网入侵检测系统的现实威胁建模和严格测量实践的必要性。



## **28. Investigating Adversarial Attacks in Software Analytics via Machine Learning Explainability**

通过机器学习解释性调查软件分析中的对抗性攻击 cs.SE

This paper has been accepted for publication in Software Quality  Journal

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2408.04124v2) [paper-pdf](http://arxiv.org/pdf/2408.04124v2)

**Authors**: MD Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: With the recent advancements in machine learning (ML), numerous ML-based approaches have been extensively applied in software analytics tasks to streamline software development and maintenance processes. Nevertheless, studies indicate that despite their potential usefulness, ML models are vulnerable to adversarial attacks, which may result in significant monetary losses in these processes. As a result, the ML models' robustness against adversarial attacks must be assessed before they are deployed in software analytics tasks. Despite several techniques being available for adversarial attacks in software analytics tasks, exploring adversarial attacks using ML explainability is largely unexplored. Therefore, this study aims to investigate the relationship between ML explainability and adversarial attacks to measure the robustness of ML models in software analytics tasks. In addition, unlike most existing attacks that directly perturb input-space, our attack approach focuses on perturbing feature-space. Our extensive experiments, involving six datasets, three ML explainability techniques, and seven ML models, demonstrate that ML explainability can be used to conduct successful adversarial attacks on ML models in software analytics tasks. This is achieved by modifying only the top 1-3 important features identified by ML explainability techniques. Consequently, the ML models under attack fail to accurately predict up to 86.6% of instances that were correctly predicted before adversarial attacks, indicating the models' low robustness against such attacks. Finally, our proposed technique demonstrates promising results compared to four state-of-the-art adversarial attack techniques targeting tabular data.

摘要: 随着机器学习（ML）的最近进步，许多基于ML的方法已被广泛应用于软件分析任务中，以简化软件开发和维护流程。然而，研究表明，尽管ML模型具有潜在的用途，但它们很容易受到对抗攻击，这可能会导致这些过程中的重大金钱损失。因此，在将ML模型部署到软件分析任务中之前，必须评估它们对对抗攻击的稳健性。尽管有多种技术可用于软件分析任务中的对抗性攻击，但使用ML解释性探索对抗性攻击在很大程度上尚未被探索。因此，本研究旨在研究ML可解释性与对抗性攻击之间的关系，以衡量ML模型在软件分析任务中的鲁棒性。此外，与大多数直接扰动输入空间的现有攻击不同，我们的攻击方法侧重于扰动特征空间。我们广泛的实验，涉及六个数据集，三个ML可解释性技术和七个ML模型，证明ML可解释性可以用于在软件分析任务中对ML模型进行成功的对抗性攻击。这是通过仅修改ML可解释性技术识别的前1-3个重要特征来实现的。因此，受攻击的ML模型无法准确预测多达86.6%的对抗性攻击之前正确预测的实例，这表明模型对此类攻击的鲁棒性较低。最后，与四种针对表格数据的最新对抗攻击技术相比，我们提出的技术表现出了有希望的结果。



## **29. BURN: Backdoor Unlearning via Adversarial Boundary Analysis**

BURN：通过对抗边界分析的后门消除学习 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10491v1) [paper-pdf](http://arxiv.org/pdf/2507.10491v1)

**Authors**: Yanghao Su, Jie Zhang, Yiming Li, Tianwei Zhang, Qing Guo, Weiming Zhang, Nenghai Yu, Nils Lukas, Wenbo Zhou

**Abstract**: Backdoor unlearning aims to remove backdoor-related information while preserving the model's original functionality. However, existing unlearning methods mainly focus on recovering trigger patterns but fail to restore the correct semantic labels of poison samples. This limitation prevents them from fully eliminating the false correlation between the trigger pattern and the target label. To address this, we leverage boundary adversarial attack techniques, revealing two key observations. First, poison samples exhibit significantly greater distances from decision boundaries compared to clean samples, indicating they require larger adversarial perturbations to change their predictions. Second, while adversarial predicted labels for clean samples are uniformly distributed, those for poison samples tend to revert to their original correct labels. Moreover, the features of poison samples restore to closely resemble those of corresponding clean samples after adding adversarial perturbations. Building upon these insights, we propose Backdoor Unlearning via adversaRial bouNdary analysis (BURN), a novel defense framework that integrates false correlation decoupling, progressive data refinement, and model purification. In the first phase, BURN employs adversarial boundary analysis to detect poisoned samples based on their abnormal adversarial boundary distances, then restores their correct semantic labels for fine-tuning. In the second phase, it employs a feedback mechanism that tracks prediction discrepancies between the original backdoored model and progressively sanitized models, guiding both dataset refinement and model purification. Extensive evaluations across multiple datasets, architectures, and seven diverse backdoor attack types confirm that BURN effectively removes backdoor threats while maintaining the model's original performance.

摘要: 后门取消学习旨在删除与后门相关的信息，同时保留模型的原始功能。然而，现有的去学习方法主要集中在恢复触发模式上，而未能恢复毒物样本的正确语义标签。这个限制阻止他们完全消除触发模式和目标标签之间的错误相关性。为了解决这个问题，我们利用边界对抗攻击技术，揭示了两个关键观察结果。首先，与干净样本相比，有毒样本与决策边界的距离显着更大，这表明它们需要更大的对抗性扰动才能改变其预测。其次，虽然清洁样本的敌对预测标签是均匀分布的，但有毒样本的标签往往会恢复到原来的正确标签。此外，添加对抗性扰动后，毒物样本的特征恢复到与相应干净样本的特征非常相似。在这些见解的基础上，我们提出了通过adversaRial bouNstival分析（BURN）的后门取消学习，这是一种新型防御框架，集成了虚假相关脱钩、渐进数据细化和模型净化。在第一阶段，BURN采用对抗边界分析，根据异常的对抗边界距离检测中毒样本，然后恢复正确的语义标签进行微调。在第二阶段，它采用了一种反馈机制，跟踪原始后门模型和逐步净化模型之间的预测差异，指导数据集细化和模型纯化。对多个数据集、架构和七种不同后门攻击类型的广泛评估证实，BURN有效地消除了后门威胁，同时保持了模型的原始性能。



## **30. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

破解LLM护栏：即时注入和越狱检测系统规避攻击的实证分析 cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六个著名的保护系统进行测试，包括微软的Azure Prompt Shield和Meta的Prompt Guard，我们表明这两种方法都可以用来逃避检测，同时保持对抗效用，在某些情况下达到100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **31. SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples**

SCOTER：无限制对抗性例子的人类评估框架 cs.CV

42 pages, 16 figures, 11 tables, Under Review, Code:  https://github.com/DrenFazlija/Scooter, Data:  https://doi.org/10.5281/zenodo.15771501

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.07776v2) [paper-pdf](http://arxiv.org/pdf/2507.07776v2)

**Authors**: Dren Fazlija, Monty-Maximilian Zühlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E. Olatunji, Daniel Kudenko

**Abstract**: Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.

摘要: 无限制的对抗攻击旨在愚弄计算机视觉模型，而不受$\ell_p$-norm边界的约束，以保持人类不可感知，例如，通过改变对象的颜色。这使得攻击者能够规避传统的、规范有限的防御策略，例如对抗性训练或认证的防御策略。然而，由于其不受限制的性质，也无法保证基于规范的不可感知性，因此需要进行人类评估来验证这些对抗性例子看起来有多真实。虽然一些相关工作评估了对抗性攻击的这一重要性质，但没有一项工作提供统计上显着的见解。这个问题需要一个统一的框架来支持和简化这种评估，以评估和比较无限制的攻击。为了缩小这一差距，我们引入了SCOOTER -一个开源的，统计动力的框架，用于评估不受限制的对抗性示例。我们的贡献是：$（i）$用于测量不可感知性的群体研究功率、补偿和李克特等效界限的最佳实践指南; $（ii）$在346名人类参与者中进行的第一次大规模人类与模型比较，表明三种颜色空间攻击和三种基于扩散的攻击无法产生不可感知的图像。此外，我们发现GPT-4 o可以作为不可感知性的初步测试，但它只能持续检测六种测试攻击中的四种攻击的对抗性示例; $（iii）$开源软件工具，包括基于浏览器的任务模板，用于收集Python和R中的注释和分析脚本;$（iv）$ImageNet衍生的基准数据集，包含3 K真实图像，7 K对抗性示例和超过34 K的人类评分。我们的研究结果表明，自动视觉系统不符合人类的感知，加强了对地面实况滑板车基准的需求。



## **32. Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach**

通过增长界矩阵方法在NLP中弥合鲁棒性和通用性对抗词替换攻击 cs.CL

Accepted to ACL Findings 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10330v1) [paper-pdf](http://arxiv.org/pdf/2507.10330v1)

**Authors**: Mohammed Bouri, Adnane Saoud

**Abstract**: Despite advancements in Natural Language Processing (NLP), models remain vulnerable to adversarial attacks, such as synonym substitutions. While prior work has focused on improving robustness for feed-forward and convolutional architectures, the robustness of recurrent networks and modern state space models (SSMs), such as S4, remains understudied. These architectures pose unique challenges due to their sequential processing and complex parameter dynamics. In this paper, we introduce a novel regularization technique based on Growth Bound Matrices (GBM) to improve NLP model robustness by reducing the impact of input perturbations on model outputs. We focus on computing the GBM for three architectures: Long Short-Term Memory (LSTM), State Space models (S4), and Convolutional Neural Networks (CNN). Our method aims to (1) enhance resilience against word substitution attacks, (2) improve generalization on clean text, and (3) providing the first systematic analysis of SSM (S4) robustness. Extensive experiments across multiple architectures and benchmark datasets demonstrate that our method improves adversarial robustness by up to 8.8% over existing baselines. These results highlight the effectiveness of our approach, outperforming several state-of-the-art methods in adversarial defense. Codes are available at https://github.com/BouriMohammed/GBM

摘要: 尽管自然语言处理（NLP）取得了进步，但模型仍然容易受到敌对攻击，例如同义词替换。虽然之前的工作重点是提高前向和卷积架构的鲁棒性，但循环网络和现代状态空间模型（SSms）（例如S4）的鲁棒性仍然研究不足。这些架构因其顺序处理和复杂的参数动态而带来独特的挑战。本文引入了一种基于增长界矩阵（GBM）的新型正规化技术，通过减少输入扰动对模型输出的影响来提高NLP模型的鲁棒性。我们专注于计算三种架构的GBM：长短期记忆（LSTM）、状态空间模型（S4）和卷积神经网络（CNN）。我们的方法旨在（1）增强针对单词替换攻击的弹性，（2）改进干净文本的概括，以及（3）提供对ESM（S4）鲁棒性的首次系统分析。跨多个架构和基准数据集的广泛实验表明，我们的方法比现有基线提高了高达8.8%的对抗稳健性。这些结果凸显了我们方法的有效性，在对抗性防御中优于几种最先进的方法。代码可访问https://github.com/BouriMohammed/GBM



## **33. Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures**

万花筒背景攻击：利用多重辐射对称纹理扰乱姿势估计 cs.CV

Accepted at ICCV 2025. Project page is available at  https://wakuwu.github.io/KBA

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10265v1) [paper-pdf](http://arxiv.org/pdf/2507.10265v1)

**Authors**: Xinlong Ding, Hongwei Yu, Jiawei Li, Feifan Li, Yu Shang, Bochao Zou, Huimin Ma, Jiansheng Chen

**Abstract**: Camera pose estimation is a fundamental computer vision task that is essential for applications like visual localization and multi-view stereo reconstruction. In the object-centric scenarios with sparse inputs, the accuracy of pose estimation can be significantly influenced by background textures that occupy major portions of the images across different viewpoints. In light of this, we introduce the Kaleidoscopic Background Attack (KBA), which uses identical segments to form discs with multi-fold radial symmetry. These discs maintain high similarity across different viewpoints, enabling effective attacks on pose estimation models even with natural texture segments. Additionally, a projected orientation consistency loss is proposed to optimize the kaleidoscopic segments, leading to significant enhancement in the attack effectiveness. Experimental results show that optimized adversarial kaleidoscopic backgrounds can effectively attack various camera pose estimation models.

摘要: 摄像机姿态估计是一项基本的计算机视觉任务，对于视觉定位和多视图立体重建等应用至关重要。在具有稀疏输入的以对象为中心的场景中，姿态估计的准确性可能会受到占据不同视角图像主要部分的背景纹理的显着影响。有鉴于此，我们引入了万花筒背景攻击（SBA），它使用相同的片段来形成具有多重辐射对称性的光盘。这些光盘在不同视角之间保持高度相似性，即使具有自然纹理片段，也能够对姿态估计模型进行有效攻击。此外，还提出了投影方向一致性损失来优化万花筒片段，从而显着提高攻击效率。实验结果表明，优化的对抗万花筒背景可以有效攻击各种摄像机姿态估计模型。



## **34. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

传输样式以减少纹理偏差并提高语义分割网络中的鲁棒性 cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10239v1) [paper-pdf](http://arxiv.org/pdf/2507.10239v1)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.

摘要: 最近的研究调查了深度神经网络（DNN）在图像分类中的形状和纹理偏差，这些偏差会影响其泛化能力和鲁棒性。研究表明，与常规DNN训练相比，使用风格化图像进行训练可以减少图像分类中的纹理偏差，并提高图像损坏的鲁棒性。为了推进这一研究，我们研究是否风格转移也可以提供这两个效果的语义分割。为此，我们进行风格转移与风格不同的人工图像区域。这些随机区域是由选定数量的Voronoi细胞形成的。然后使用生成的风格传输数据来训练语义分割DNN，目标是减少它们对纹理线索的依赖，同时增强它们对基于形状的特征的依赖。在我们的实验中，事实证明，在语义分割中，风格转移增强减少了纹理偏差，并大大提高了针对常见图像损坏和对抗攻击的鲁棒性。这些观察结果适用于Cityscapes数据集以及PASCAL Context上的卷积神经网络和Transformer架构，显示了所提出方法的通用性。



## **35. HASSLE: A Self-Supervised Learning Enhanced Hijacking Attack on Vertical Federated Learning**

HASSLE：自我监督学习对垂直联邦学习的增强劫持攻击 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10162v1) [paper-pdf](http://arxiv.org/pdf/2507.10162v1)

**Authors**: Weiyang He, Chip-Hong Chang

**Abstract**: Vertical Federated Learning (VFL) enables an orchestrating active party to perform a machine learning task by cooperating with passive parties that provide additional task-related features for the same training data entities. While prior research has leveraged the privacy vulnerability of VFL to compromise its integrity through a combination of label inference and backdoor attacks, their effectiveness is constrained by the low label inference precision and suboptimal backdoor injection conditions. To facilitate a more rigorous security evaluation on VFL without these limitations, we propose HASSLE, a hijacking attack framework composed of a gradient-direction-based label inference module and an adversarial embedding generation algorithm enhanced by self-supervised learning. HASSLE accurately identifies private samples associated with a targeted label using only a single known instance of that label. In the two-party scenario, it demonstrates strong performance with an attack success rate (ASR) of over 99% across four datasets, including both image and tabular modalities, and achieves 85% ASR on the more complex CIFAR-100 dataset. Evaluation of HASSLE against 8 potential defenses further highlights its significant threat while providing new insights into building a trustworthy VFL system.

摘要: 垂直联合学习（VFL）使协调主动方能够通过与被动方合作来执行机器学习任务，被动方为相同的训练数据实体提供额外的任务相关功能。虽然之前的研究利用VFL的隐私漏洞通过标签推断和后门攻击的组合来损害其完整性，但它们的有效性受到标签推断精度低和次优后门注入条件的限制。为了在没有这些限制的情况下对VFL进行更严格的安全评估，我们提出了HASSLE，这是一个劫持攻击框架，由基于梯度方向的标签推理模块和通过自我监督学习增强的对抗嵌入生成算法组成。HASSLE仅使用该标签的单个已知实例来准确识别与目标标签相关的私人样本。在双方场景中，它表现出强大的性能，在四个数据集（包括图像和表格模式）中攻击成功率（ASB）超过99%，并在更复杂的CIFAR-100数据集上实现了85%的ASB。针对8种潜在防御措施对HASSLE进行的评估进一步凸显了其重大威胁，同时为构建值得信赖的VFL系统提供了新的见解。



## **36. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

使用LLM的显式漏洞生成：对抗性攻击之外的调查 cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10054v1) [paper-pdf](http://arxiv.org/pdf/2507.10054v1)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.

摘要: 大型语言模型（LLM）越来越多地被用作代码助手，但当被明确要求生成不安全代码时，它们的行为仍然知之甚少。虽然之前的研究重点是无意的漏洞或对抗性提示技术，但这项研究考察了更直接的威胁场景：开源LLM在直接或间接提示时生成易受攻击的代码。我们提出了一种双重实验设计：（1）动态预算处理，系统性地改变结构化模板中的漏洞类型、用户个性和直接性;（2）反向预算处理，提示从真实的漏洞代码样本中获取来评估漏洞复制的准确性。我们使用ESBMC静态分析评估了三个开源7 B参数模型（Qwen 2、Mistral和Gemma），以评估漏洞的存在和生成的漏洞类型的正确性。结果显示，所有模型经常产生脆弱的输出，其中Qwen 2实现了最高的正确率。用户角色对成功有着显着的影响，学生角色的脆弱性率高于专业角色，而直接提示则稍微有效一些。脆弱性复制遵循倒U型模式，具有圈复杂性，在中等范围内达到峰值。我们的研究结果揭示了开源模型中安全机制的局限性，特别是对于看似良性的教育请求。



## **37. 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving**

3DGAA：针对自动驾驶的真实且稳健的3D基于高斯的对抗攻击 cs.CV

Submitted to WACV 2026

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09993v1) [paper-pdf](http://arxiv.org/pdf/2507.09993v1)

**Authors**: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

**Abstract**: Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving.

摘要: 基于摄像头的物体检测系统在自动驾驶中发挥着至关重要的作用，但它们在现实世界环境中仍然容易受到对抗威胁的影响。虽然现有的2D和3D物理攻击通常会优化纹理，但它们通常很难平衡物理真实感和攻击鲁棒性。在这项工作中，我们提出了基于3D高斯的对抗性攻击（3DGAA），这是一种新型的对抗性对象生成框架，它利用3D高斯飞溅（3DGS）的完整14维参数化来以物理可实现的方式联合优化几何形状和外观。与之前依赖补丁或纹理的作品不同，3DGAA联合扰动几何属性（形状、比例、旋转）和外观属性（颜色、不透明度），以产生物理上真实且可转移的对抗对象。我们进一步引入物理过滤模块来保持几何保真度，并引入物理增强模块来模拟复杂的物理场景，从而增强现实世界条件下的攻击概括性。我们使用微型车辆模型在虚拟基准和物理世界设置上评估3DGAA。实验结果表明，3DGAA将检测mAP从87.21%降低到7.38%，显着优于现有的3D物理攻击。此外，我们的方法在不同的物理条件下保持了高度的可移植性，展示了物理可实现的对抗攻击的新技术水平。这些结果验证了3DGAA作为评估自动驾驶中感知系统安全性的实用攻击框架。



## **38. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.03940v3) [paper-pdf](http://arxiv.org/pdf/2501.03940v3)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型（LLM）的快速发展显着增强了它们生成连贯且上下文相关文本的能力，引发了人们对人工智能生成内容滥用的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在未知的领域或不熟悉的LLM中。利用LLM下一个代币分发输出提供了一种理论上有吸引力的检测方法，因为它们包含了模型对不同数据库进行的广泛预训练的见解。尽管有希望，但试图实现这些输出的零射击方法收效有限。我们假设问题之一是，他们使用平均值来汇总各个代币之间的下一个代币分布指标，而有些代币自然更容易或更难预测，并且应该采用不同的加权方式。基于这个想法，我们提出了困惑注意力加权网络（PAWN），它使用LLM的最后隐藏状态和位置来根据整个序列长度的下一个令牌分布的指标对一系列特征的总和进行加权。尽管不是零射击，但我们的方法允许我们在磁盘上缓存最后一个隐藏状态和下一个令牌分布指标，从而大大减少了训练资源需求。PAWN在可训练参数仅为一小部分的情况下表现出比最强基线（微调LM）有竞争力甚至更好的分布性能。我们的模型还可以更好地推广到不可见的域和源模型，分布变化中决策边界的变异性较小。它对对抗攻击也更稳健，如果主干具有多语言能力，它会对监督训练期间未见过的语言进行良好的概括，LLaMA 3 -1B在与九种语言的交叉验证中达到了81.46%的平均宏平均F1得分。



## **39. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

EVALOOP：从自我一致性的角度评估LLM编程稳健性 cs.SE

20 pages, 11 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2505.12185v3) [paper-pdf](http://arxiv.org/pdf/2505.12185v3)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.

摘要: 评估大型语言模型（LLM）的编程能力对于它们在软件工程中的有效使用至关重要。然而，当前的评估主要衡量静态基准上生成的代码的准确性，忽视了编程任务期间模型稳健性的关键方面。虽然对抗性攻击提供了有关模型稳健性的见解，但它们的有效性有限，并且评估可能会受到限制。当前用于稳健性评估的对抗攻击方法会产生不一致的结果，难以在不同的LLM之间提供统一的评估。我们引入EVALOOP，这是一种新型评估框架，从自一致性的角度评估稳健性，即利用流行软件工程任务中固有的自然二重性，例如，代码生成和代码摘要。EVALOOP启动独立反馈循环：LLM生成输出（例如，代码）来自输入（例如，自然语言规范），然后使用生成的输出作为输入来产生新的输出（例如，将该代码总结为新规范）。EVALOOP重复该过程以评估每个循环中EVALOOP的有效性。这种循环策略本质上评估稳健性，而不依赖任何外部攻击设置，提供了一个统一的指标来评估LLM在编程中的稳健性。我们评估了16个著名的LLM（例如，GPT-4.1，O 4-mini）在EVALOOP上发现EVALOOP通常会在十个循环内导致pass@1性能绝对下降5.01%-19.31%。有趣的是，稳健性并不总是与初始性能一致（即，一次性查询）;例如，GPT-3.5-Turbo尽管初始代码生成优于DeepSeek-V2，但在重复评估循环中表现出较低的鲁棒性。



## **40. AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective**

AdvGrasp：从物理角度对机器人抓取的对抗性攻击 cs.RO

IJCAI'2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09857v1) [paper-pdf](http://arxiv.org/pdf/2507.09857v1)

**Authors**: Xiaofei Wang, Mingliang Han, Tianyu Hao, Cegang Li, Yunbo Zhao, Keke Tang

**Abstract**: Adversarial attacks on robotic grasping provide valuable insights into evaluating and improving the robustness of these systems. Unlike studies that focus solely on neural network predictions while overlooking the physical principles of grasping, this paper introduces AdvGrasp, a framework for adversarial attacks on robotic grasping from a physical perspective. Specifically, AdvGrasp targets two core aspects: lift capability, which evaluates the ability to lift objects against gravity, and grasp stability, which assesses resistance to external disturbances. By deforming the object's shape to increase gravitational torque and reduce stability margin in the wrench space, our method systematically degrades these two key grasping metrics, generating adversarial objects that compromise grasp performance. Extensive experiments across diverse scenarios validate the effectiveness of AdvGrasp, while real-world validations demonstrate its robustness and practical applicability

摘要: 对机器人抓取的对抗攻击为评估和提高这些系统的稳健性提供了宝贵的见解。与仅关注神经网络预测而忽视抓取物理原理的研究不同，本文引入了AdvGrasp，这是一个从物理角度对机器人抓取进行对抗攻击的框架。具体来说，AdvGrasp的目标是两个核心方面：提升能力（评估抵抗重力提升物体的能力）和抓取稳定性（评估抵抗外部干扰的能力）。通过变形物体的形状以增加重力扭矩并减少扳手空间中的稳定性裕度，我们的方法系统性地降低了这两个关键抓取指标，生成损害抓取性能的对抗物体。跨不同场景的广泛实验验证了AdvGrasp的有效性，而现实世界的验证则证明了其稳健性和实际适用性



## **41. Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats**

博弈论与法学硕士和抽象人工智能相遇：为智能威胁时代重新构想网络安全 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10621v1) [paper-pdf](http://arxiv.org/pdf/2507.10621v1)

**Authors**: Quanyan Zhu

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.   The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.   This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.

摘要: 保护网络空间不仅需要先进的工具，还需要改变我们对威胁、信任和自主性的推理方式。传统的网络安全方法依赖于手动响应和脆弱的启发式方法。为了构建主动和智能的防御系统，我们需要集成的理论框架和软件工具。博弈论为对抗行为建模、设计战略防御和实现自治系统信任提供了严格的基础。与此同时，软件工具处理网络数据、可视化攻击表面、验证合规性并建议缓解措施。然而，理论和实际实施之间仍然存在脱节。   大型语言模型（LLM）和代理人工智能的兴起为弥合这一差距提供了一条新的途径。LLM支持的代理可以将抽象策略实施为现实世界的决策。相反，博弈论可以为这些代理在复杂工作流程中的推理和协调提供信息。LLM还挑战了经典的博弈论假设，例如完美理性或静态收益，促使新模型与认知和计算现实保持一致。这种协同进化提供了更丰富的理论基础和新的解决方案概念。人工智能还重塑了软件设计：系统现在必须从一开始就具有模块化、自适应性和信任意识。   本章探讨博弈论、代理人工智能和网络安全的交叉点。我们回顾了关键的博弈论框架（例如，静态、动态、Bayesian和Signal Game）和解决方案概念。然后，我们研究LLM代理如何增强网络防御并引入LLM驱动的将推理嵌入AI代理的游戏。最后，我们探索多代理工作流程和协调游戏，概述了这种融合如何培养安全、智能和自适应的网络系统。



## **42. Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations**

概念掌舵者：利用K稀疏自动编码器实现测试时间可控生成 cs.CV

23 pages, 18 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.19066v2) [paper-pdf](http://arxiv.org/pdf/2501.19066v2)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers

摘要: 尽管文本到图像生成模型取得了显着的进步，但它们容易受到对抗性攻击，并无意中生成不安全、不道德的内容。现有的方法通常依赖于微调模型来删除特定概念，这在计算上昂贵、缺乏可扩展性和/或损害发电质量。在这项工作中，我们提出了一种利用k-稀疏自动编码器（k-SAEs）的新型框架，以在扩散模型中实现高效且可解释的概念操纵。具体来说，我们首先在文本嵌入的潜在空间中识别可解释的单语义概念，并利用它们来精确地引导生成远离或转向给定概念（例如，裸体）或引入新概念（例如，摄影风格）--所有在测试时间。通过大量实验，我们证明我们的方法非常简单，不需要重新训练基本模型或LoRA适配器，不损害生成质量，并且对对抗提示操作具有鲁棒性。我们的方法在不安全概念删除方面提高了$\mathBF{20.01\%}$，在风格操纵方面有效，并且比当前最先进技术快$\mathBF{\sim 5}$x。代码可访问：https://github.com/kim-dahye/steerers



## **43. Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers**

对抗激活修补：检测和减轻安全调整变形金刚中紧急欺骗的框架 cs.LG

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09406v1) [paper-pdf](http://arxiv.org/pdf/2507.09406v1)

**Authors**: Santhosh Kumar Ravindran

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models.

摘要: 通过人类反馈强化学习（RL HF）等技术实现安全性调整的大型语言模型（LLM）通常表现出紧急欺骗行为，其中输出看起来合规，但微妙地误导或省略关键信息。本文介绍了对抗性激活补丁，这是一种新型的机械解释性框架，它利用激活补丁作为对抗性工具来诱导、检测和减轻基于转换器的模型中的此类欺骗。通过从“欺骗性”提示中获取激活并将其修补为特定层的安全转发传递，我们模拟漏洞并量化欺骗率。通过跨多个场景的玩具神经网络模拟（例如，每个设置1000次试验），我们证明对抗性修补将欺骗性输出从0%基线增加到23.9%，特定层的变化支持我们的假设。我们提出了六个假设，包括模型之间的可移植性、多模式环境的恶化以及缩放效应。扩大的文献评论综合了可解释性、欺骗性和对抗性攻击方面的20多部关键作品。详细介绍了缓解策略，例如激活异常检测和稳健的微调，以及道德考虑和未来的研究方向。这项工作通过强调修补的双重用途潜力来提高人工智能安全性，并为大规模模型的实证研究提供了路线图。



## **44. Single Word Change is All You Need: Using LLMs to Create Synthetic Training Examples for Text Classifiers**

更改单个单词即可：使用LLM为文本分类器创建合成训练示例 cs.CL

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2401.17196v3) [paper-pdf](http://arxiv.org/pdf/2401.17196v3)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric $\rho$ to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves $\rho$ by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.

摘要: 在文本分类中，创建对抗性示例意味着微妙地扰乱句子中的几个词而不改变其含义，导致其被分类器错误分类。一个令人担忧的观察是，现有方法生成的很大一部分对抗性例子只改变了一个词。这种单字扰动漏洞代表了分类器的一个重大弱点，恶意用户可以利用它有效地创建大量对抗性示例。本文研究了这个问题并做出了以下关键贡献：（1）我们引入了一种新型指标$\rho$来定量评估分类器对单字扰动的鲁棒性。(2)我们提出了SP-Attack，旨在利用单字扰动漏洞，实现更高的攻击成功率，更好地保留句子含义，同时与最先进的对抗方法相比降低了计算成本。(3)我们提出SP-Defense，旨在通过在学习中应用数据增强来改进\r{ho}。对4个数据集以及BERT和DistilBERT分类器的实验结果表明，SP-Defense将$\rho$提高了14.6%和13.9%，并将SP-Attack在两个分类器上的攻击成功率分别降低了30.4%和21.2%，并降低了现有涉及多字扰动的攻击方法的攻击成功率。



## **45. AdRo-FL: Informed and Secure Client Selection for Federated Learning in the Presence of Adversarial Aggregator**

AdRo-FL：在对抗性聚合器的存在下为联邦学习进行明智且安全的客户端选择 cs.CR

17 pages

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2506.17805v2) [paper-pdf](http://arxiv.org/pdf/2506.17805v2)

**Authors**: Md. Kamrul Hossain, Walid Aljoby, Anis Elgabli, Ahmed M. Abdelmoniem, Khaled A. Harras

**Abstract**: Federated Learning (FL) enables collaborative learning without exposing clients' data. While clients only share model updates with the aggregator, studies reveal that aggregators can infer sensitive information from these updates. Secure Aggregation (SA) protects individual updates during transmission; however, recent work demonstrates a critical vulnerability where adversarial aggregators manipulate client selection to bypass SA protections, constituting a Biased Selection Attack (BSA). Although verifiable random selection prevents BSA, it precludes informed client selection essential for FL performance. We propose Adversarial Robust Federated Learning (AdRo-FL), which simultaneously enables: informed client selection based on client utility, and robust defense against BSA maintaining privacy-preserving aggregation. AdRo-FL implements two client selection frameworks tailored for distinct settings. The first framework assumes clients are grouped into clusters based on mutual trust, such as different branches of an organization. The second framework handles distributed clients where no trust relationships exist between them. For the cluster-oriented setting, we propose a novel defense against BSA by (1) enforcing a minimum client selection quota from each cluster, supervised by a cluster-head in every round, and (2) introducing a client utility function to prioritize efficient clients. For the distributed setting, we design a two-phase selection protocol: first, the aggregator selects the top clients based on our utility-driven ranking; then, a verifiable random function (VRF) ensures a BSA-resistant final selection. AdRo-FL also applies quantization to reduce communication overhead and sets strict transmission deadlines to improve energy efficiency. AdRo-FL achieves up to $1.85\times$ faster time-to-accuracy and up to $1.06\times$ higher final accuracy compared to insecure baselines.

摘要: 联合学习（FL）可在不暴露客户数据的情况下实现协作学习。虽然客户端仅与聚合器共享模型更新，但研究表明，聚合器可以从这些更新中推断敏感信息。安全聚合（SA）在传输过程中保护各个更新;然而，最近的工作表明了一个严重的漏洞，其中对抗性聚合器操纵客户端选择以绕过SA保护，从而构成偏向选择攻击（BSA）。尽管可验证的随机选择会阻止BSA，但它会阻止对FL性能至关重要的知情客户选择。我们提出了对抗性鲁棒联邦学习（AdRo-FL），它同时能够：根据客户机效用进行明智的客户机选择，以及针对BSA的稳健防御，以维持隐私保护聚合。AdRo-FL实现了两个针对不同设置量身定制的客户选择框架。第一个框架假设客户根据相互信任分组到集群中，例如组织的不同分支机构。第二个框架处理之间不存在信任关系的分布式客户端。对于面向集群的设置，我们提出了一种针对BSA的新颖防御方法，方法是：（1）在每一轮由集群头监督，从每个集群强制执行最低客户端选择配额，以及（2）引入客户端效用函数来确定高效客户端的优先顺序。对于分布式环境，我们设计了两阶段选择协议：首先，聚合器根据我们的实用程序驱动排名选择顶级客户端;然后，可验证随机函数（VRF）确保最终选择抗BSA。AdRo-FL还应用量化来减少通信负担，并设定严格的传输截止日期以提高能源效率。与不安全的基线相比，AdRo-FL的准确性时间提高了1.85美元，最终准确性提高了1.06美元。



## **46. Exploiting Leaderboards for Large-Scale Distribution of Malicious Models**

利用排行榜进行恶意模型的大规模分发 cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08983v1) [paper-pdf](http://arxiv.org/pdf/2507.08983v1)

**Authors**: Anshuman Suri, Harsh Chaudhari, Yuefeng Peng, Ali Naseh, Amir Houmansadr, Alina Oprea

**Abstract**: While poisoning attacks on machine learning models have been extensively studied, the mechanisms by which adversaries can distribute poisoned models at scale remain largely unexplored. In this paper, we shed light on how model leaderboards -- ranked platforms for model discovery and evaluation -- can serve as a powerful channel for adversaries for stealthy large-scale distribution of poisoned models. We present TrojanClimb, a general framework that enables injection of malicious behaviors while maintaining competitive leaderboard performance. We demonstrate its effectiveness across four diverse modalities: text-embedding, text-generation, text-to-speech and text-to-image, showing that adversaries can successfully achieve high leaderboard rankings while embedding arbitrary harmful functionalities, from backdoors to bias injection. Our findings reveal a significant vulnerability in the machine learning ecosystem, highlighting the urgent need to redesign leaderboard evaluation mechanisms to detect and filter malicious (e.g., poisoned) models, while exposing broader security implications for the machine learning community regarding the risks of adopting models from unverified sources.

摘要: 虽然对机器学习模型的中毒攻击已经得到了广泛的研究，但对手大规模分发中毒模型的机制在很大程度上仍未被探索。在本文中，我们揭示了模型排行榜（模型发现和评估的排名平台）如何成为对手秘密大规模分发有毒模型的强大渠道。我们介绍了TrojanClimb，这是一个通用框架，可以注入恶意行为，同时保持有竞争力的排行榜性能。我们展示了它在四种不同模式下的有效性：文本嵌入、文本生成、文本到语音和文本到图像，表明对手可以成功地获得很高的排行榜排名，同时嵌入任意有害功能（从后门到偏见注入）。我们的研究结果揭示了机器学习生态系统中的一个重大漏洞，凸显了重新设计排行榜评估机制以检测和过滤恶意（例如，中毒）模型，同时暴露了采用未经验证来源的模型的风险给机器学习社区带来的更广泛的安全影响。



## **47. VIP: Visual Information Protection through Adversarial Attacks on Vision-Language Models**

VIP：通过对视觉语言模型的对抗性攻击保护视觉信息 eess.IV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08982v1) [paper-pdf](http://arxiv.org/pdf/2507.08982v1)

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Déforges

**Abstract**: Recent years have witnessed remarkable progress in developing Vision-Language Models (VLMs) capable of processing both textual and visual inputs. These models have demonstrated impressive performance, leading to their widespread adoption in various applications. However, this widespread raises serious concerns regarding user privacy, particularly when models inadvertently process or expose private visual information. In this work, we frame the preservation of privacy in VLMs as an adversarial attack problem. We propose a novel attack strategy that selectively conceals information within designated Region Of Interests (ROIs) in an image, effectively preventing VLMs from accessing sensitive content while preserving the semantic integrity of the remaining image. Unlike conventional adversarial attacks that often disrupt the entire image, our method maintains high coherence in unmasked areas. Experimental results across three state-of-the-art VLMs namely LLaVA, Instruct-BLIP, and BLIP2-T5 demonstrate up to 98% reduction in detecting targeted ROIs, while maintaining global image semantics intact, as confirmed by high similarity scores between clean and adversarial outputs. We believe that this work contributes to a more privacy conscious use of multimodal models and offers a practical tool for further research, with the source code publicly available at: https://github.com/hbrachemi/Vlm_defense-attack.

摘要: 近年来，在开发能够处理文本和视觉输入的视觉语言模型（VLM）方面取得了显着进展。这些模型表现出令人印象深刻的性能，导致它们在各种应用中广泛采用。然而，这种广泛的情况引发了人们对用户隐私的严重担忧，特别是当模型无意中处理或暴露私人视觉信息时。在这项工作中，我们将VLM中的隐私保护定义为一个对抗性攻击问题。我们提出了一种新颖的攻击策略，可以选择性地隐藏图像中指定兴趣区域（ROI）内的信息，有效地防止VLM访问敏感内容，同时保留剩余图像的语义完整性。与通常破坏整个图像的传统对抗攻击不同，我们的方法在未掩蔽区域保持了高度一致性。三种最先进的VLM（LLaVA、INSTIT-BLIP和BLIP 2-T5）的实验结果表明，在检测目标兴趣区方面可减少高达98%，同时保持全局图像语义完整，这一点由干净和对抗性输出之间的高相似性得分所证实。我们相信这项工作有助于更加注重隐私地使用多模式模型，并为进一步研究提供了实用的工具，源代码可在https://github.com/hbrachemi/Vlm_defense-attack上公开获取。



## **48. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **49. Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security**

纠缠威胁：量子机器学习安全的统一杀死链模型 quant-ph

Accepted for publication at IEEE International Conference on Quantum  Computing and Engineering (QCE) 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08623v1) [paper-pdf](http://arxiv.org/pdf/2507.08623v1)

**Authors**: Pascal Debus, Maximilian Wendlinger, Kilian Tscharke, Daniel Herr, Cedric Brügmann, Daniel Ohl de Mello, Juris Ulmanis, Alexander Erhard, Arthur Schmidt, Fabian Petsch

**Abstract**: Quantum Machine Learning (QML) systems inherit vulnerabilities from classical machine learning while introducing new attack surfaces rooted in the physical and algorithmic layers of quantum computing. Despite a growing body of research on individual attack vectors - ranging from adversarial poisoning and evasion to circuit-level backdoors, side-channel leakage, and model extraction - these threats are often analyzed in isolation, with unrealistic assumptions about attacker capabilities and system environments. This fragmentation hampers the development of effective, holistic defense strategies. In this work, we argue that QML security requires more structured modeling of the attack surface, capturing not only individual techniques but also their relationships, prerequisites, and potential impact across the QML pipeline. We propose adapting kill chain models, widely used in classical IT and cybersecurity, to the quantum machine learning context. Such models allow for structured reasoning about attacker objectives, capabilities, and possible multi-stage attack paths - spanning reconnaissance, initial access, manipulation, persistence, and exfiltration. Based on extensive literature analysis, we present a detailed taxonomy of QML attack vectors mapped to corresponding stages in a quantum-aware kill chain framework that is inspired by the MITRE ATLAS for classical machine learning. We highlight interdependencies between physical-level threats (like side-channel leakage and crosstalk faults), data and algorithm manipulation (such as poisoning or circuit backdoors), and privacy attacks (including model extraction and training data inference). This work provides a foundation for more realistic threat modeling and proactive security-in-depth design in the emerging field of quantum machine learning.

摘要: 量子机器学习（QML）系统继承了经典机器学习的漏洞，同时引入了植根于量子计算物理和算法层的新攻击表面。尽管对个体攻击载体的研究越来越多--从对抗性中毒和规避到电路级后门、侧通道泄漏和模型提取--但这些威胁通常是孤立地分析的，对攻击者的能力和系统环境做出了不切实际的假设。这种碎片化阻碍了有效、整体防御战略的制定。在这项工作中，我们认为QML安全需要对攻击表面进行更结构化的建模，不仅捕获单个技术，还捕获它们的关系、先决条件和整个QML管道的潜在影响。我们建议将广泛用于经典IT和网络安全的杀死链模型适应量子机器学习环境。此类模型允许对攻击者的目标、能力和可能的多阶段攻击路径进行结构化推理--跨越侦察、初始访问、操纵、持久性和溢出。基于广泛的文献分析，我们提出了映射到量子感知杀死链框架中相应阶段的QML攻击载体的详细分类，该框架的灵感来自经典机器学习的MITRE ATLAS。我们强调物理级别威胁（例如侧通道泄漏和串话故障）、数据和算法操纵（例如中毒或电路后门）以及隐私攻击（包括模型提取和训练数据推断）之间的相互依赖性。这项工作为量子机器学习新兴领域更现实的威胁建模和主动安全深度设计提供了基础。



## **50. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



