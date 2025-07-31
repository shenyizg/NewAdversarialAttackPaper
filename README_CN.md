# Latest Adversarial Attack Papers
**update at 2025-07-31 10:32:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS**

AUV-Fusion：针对VARS的用户交互和视觉扰动的跨模态对抗性融合 cs.IR

14 pages,6 figures

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22880v1) [paper-pdf](http://arxiv.org/pdf/2507.22880v1)

**Authors**: Hai Ling, Tianchi Wang, Xiaohao Liu, Zhulin Tao, Lifang Yang, Xianglin Huang

**Abstract**: Modern Visual-Aware Recommender Systems (VARS) exploit the integration of user interaction data and visual features to deliver personalized recommendations with high precision. However, their robustness against adversarial attacks remains largely underexplored, posing significant risks to system reliability and security. Existing attack strategies suffer from notable limitations: shilling attacks are costly and detectable, and visual-only perturbations often fail to align with user preferences. To address these challenges, we propose AUV-Fusion, a cross-modal adversarial attack framework that adopts high-order user preference modeling and cross-modal adversary generation. Specifically, we obtain robust user embeddings through multi-hop user-item interactions and transform them via an MLP into semantically aligned perturbations. These perturbations are injected onto the latent space of a pre-trained VAE within the diffusion model. By synergistically integrating genuine user interaction data with visually plausible perturbations, AUV-Fusion eliminates the need for injecting fake user profiles and effectively mitigates the challenge of insufficient user preference extraction inherent in traditional visual-only attacks. Comprehensive evaluations on diverse VARS architectures and real-world datasets demonstrate that AUV-Fusion significantly enhances the exposure of target (cold-start) items compared to conventional baseline methods. Moreover, AUV-Fusion maintains exceptional stealth under rigorous scrutiny.

摘要: 现代视觉感知推荐系统（VAR）利用用户交互数据和视觉特征的集成来提供高精度的个性化推荐。然而，它们对对抗攻击的鲁棒性在很大程度上仍未得到充分开发，这对系统的可靠性和安全性构成了重大风险。现有的攻击策略存在明显的局限性：先令攻击成本高昂且可检测，并且仅视觉干扰通常无法与用户偏好保持一致。为了应对这些挑战，我们提出了AUV-Fusion，这是一种跨模式对抗攻击框架，采用高级用户偏好建模和跨模式对手生成。具体来说，我们通过多跳用户项交互来获得稳健的用户嵌入，并通过MLP将它们转换为语义对齐的扰动。这些扰动被注入到扩散模型内预训练的VAE的潜在空间中。通过将真实的用户交互数据与视觉上合理的干扰协同集成，AUV-Fusion消除了注入虚假用户配置文件的需要，并有效地缓解了传统纯视觉攻击中固有的用户偏好提取不足的挑战。对各种VAR架构和现实世界数据集的全面评估表明，与传统基线方法相比，AUV-Fusion显着增强了目标（冷启动）物品的暴露。此外，AUV-Fusion在严格审查下保持了出色的隐身性。



## **2. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

弯曲动态黑匣子攻击：通过动态弯曲估计重新审视对抗鲁棒性 cs.LG

This article contains several flaws

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.19194v2) [paper-pdf](http://arxiv.org/pdf/2505.19194v2)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.

摘要: 对抗性攻击揭示了深度学习模型的脆弱性。大约十年来，人们提出了无数的攻击和防御方法，从而产生了鲁棒化分类器并更好地理解模型。在这些方法中，基于弯曲的方法引起了人们的关注，因为人们认为高弯曲可能会产生粗略的决策边界。然而，最常用的\textit{currency}是模型内的损失函数、分数或其他参数的弯曲，而不是决策边界弯曲，因为前者可以相对容易地使用二阶求导形成。在本文中，我们提出了一种新的查询高效方法--动态弯曲估计（VCE），来估计黑匣子环境下的决策边界弯曲。我们的方法基于CGBA，这是一种黑匣子对抗攻击。通过对广泛的分类器执行VCE，我们从统计上发现了决策边界弯曲和对抗鲁棒性之间的联系。我们还提出了一种新的攻击方法：弯曲动态黑匣子攻击（CDBA），使用动态估计的弯曲来提高性能。



## **3. DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion**

Distill：通过潜在扩散对可疑特洛伊木马输入进行无数据倒置 cs.CV

ICCV 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22813v1) [paper-pdf](http://arxiv.org/pdf/2507.22813v1)

**Authors**: Hossein Mirzaei, Zeinab Taghavi, Sepehr Rezaee, Masoud Hadi, Moein Madadi, Mackenzie W. Mathis

**Abstract**: Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL.

摘要: 深度神经网络在众多任务中取得了显着的成功，但它们仍然容易受到特洛伊木马（后门）攻击，这引发了人们对其在现实世界任务关键型应用中安全性的严重担忧。常见的对策是触发倒置--重建对手在训练期间插入的恶意“快捷”模式（触发器）。当前的触发器倒置方法通常在特定假设下搜索整个像素空间，但不能保证估计的触发不仅仅是翻转模型输出的对抗性扰动。在这里，我们提出了一种无数据、零触发触发器倒置策略，该策略限制了搜索空间，同时避免了对触发器外观的强烈假设。具体来说，我们结合了一个由目标分类器引导的基于扩散的生成器;通过迭代生成，我们生成与模型所依赖的恶意行为的内部表示一致的候选触发器。定量和定性的经验评估表明，我们的方法重建了触发器，可以有效区分干净模型和特洛伊模型。Distill以很高的优势超越了替代方法，在BackdoorBench数据集上实现了高达7.1%的准确性，在木马对象检测模型扫描上提高了9.4%，为可靠的后门防御提供了一个有希望的新方向，而无需依赖大量数据或对触发器的强大先验假设。该代码可在https://github.com/AdaptiveMotorControlLab/DISTIL上获取。



## **4. Cryptanalysis of LC-MUME: A Lightweight Certificateless Multi-User Matchmaking Encryption for Mobile Devices**

LC-MUME的加密分析：一种用于移动设备的轻量级无证书多用户匹配加密 cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22674v1) [paper-pdf](http://arxiv.org/pdf/2507.22674v1)

**Authors**: Ramprasad Sarkar

**Abstract**: Yang et al. proposed a lightweight certificateless multiuser matchmaking encryption (LC-MUME) scheme for mobile devices, published in IEEE Transactions on Information Forensics and Security (TIFS) (DOI: 10.1109/TIFS.2023.3321961). Their construction aims to reduce computational and communication overhead within a one-to-many certificateless cryptographic framework. The authors claim that their scheme satisfies existential unforgeability under chosen-message attacks (EUF-CMA) in the random oracle model. However, our cryptanalytic study demonstrates that the scheme fails to meet this critical security requirement. In particular, we show that a Type-I adversary can successfully forge a valid ciphertext without possessing the complete private key of the sender. Both theoretical analysis and practical implementation confirm that this attack can be mounted with minimal computational cost. To address these weaknesses, we propose a modification strategy to strengthen the security of matchmaking encryption schemes in mobile computing environments.

摘要: Yang等人提出了一种针对移动设备的轻量级无证书多用户匹配加密（LC-MUME）方案，发表在《IEEE信息取证与安全交易》（TIFS）（DOI：10.1109/TIFS.2023.3321961）中。他们的构建旨在减少一对多无证书加密框架内的计算和通信负担。作者声称，他们的方案在随机预言模型中满足选择消息攻击（EUF-CMA）下的存在不可伪造性。然而，我们的密码分析研究表明，该计划未能满足这一关键的安全要求。特别是，我们表明I型对手可以在不拥有发送者完整的私有密钥的情况下成功伪造有效的密文。理论分析和实际实现都证实，这种攻击可以以最小的计算成本发起。为了解决这些弱点，我们提出了一种修改策略来加强移动计算环境中匹配加密方案的安全性。



## **5. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动对视觉系统构成重大威胁。传统的防御方法通常需要重新训练或微调，这使得它们在实际部署中不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了视觉语言模型（VLM）用于对抗性补丁检测。通过在不断扩展的数据库中检索与存储的攻击相似的视觉上相似的补丁和图像，VRAG执行生成推理以识别不同的攻击类型，所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **6. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

基于扩散的对抗性身份操纵用于面部隐私保护 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.21646v3) [paper-pdf](http://arxiv.org/pdf/2504.21646v3)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.

摘要: 由于社交网络上潜在的未经授权的监视和用户跟踪，面部识别（FR）系统的成功引发了严重的隐私问题。现有的增强隐私的方法无法生成可以保护面部隐私的自然面部图像。在本文中，我们提出了基于扩散的对抗身份操纵（DiffAIM）来生成针对恶意FR系统的自然且高度可转移的对抗面孔。具体来说，我们在扩散模型的低维潜在空间内操纵面部身份。这涉及在反向扩散过程中迭代地注入基于梯度的对抗性身份指导，逐步引导一代人走向所需的对抗性面孔。该指南针对向目标的身份融合进行了优化，同时促进源自源头的语义分歧，促进有效模仿，同时保持视觉自然性。我们进一步结合了结构保留的正规化，以在操作过程中保持面部结构一致性。针对人脸验证和识别任务的大量实验表明，与最新技术相比，迪夫AIM实现了更强的黑匣子攻击可转移性，同时保持了卓越的视觉质量。我们还证明了所提出的方法对商业FR API（包括Face++和Aliyun）的有效性。



## **7. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **8. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

使用具有指定概率操纵的白盒对抗攻击对DNN模型进行所有权验证 cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.17579v3) [paper-pdf](http://arxiv.org/pdf/2505.17579v3)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Isobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.

摘要: 在本文中，我们提出了一种新的框架，用于图像分类任务的深度神经网络（DNN）模型的所有权验证。它允许合法所有者和第三方验证型号身份，而无需出示原始型号。我们假设一个灰盒场景，其中未经授权的用户拥有从原始模型非法复制的模型，在云环境中提供服务，用户抛出图像并接收分类结果作为输出类的概率分布。该框架应用白盒对抗攻击，将特定类的输出概率与指定值对齐。由于对原始模型的了解，它使所有者能够生成此类对抗性示例。我们通过引入控制参数，提出了一种基于迭代快速梯度符号法（FGSM）的简单但有效的对抗攻击方法。实验结果证实了使用对抗攻击识别DNN模型的有效性。



## **9. RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**

RCR-AF：通过Rademacher复杂性降低激活函数增强模型概括 cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22446v1) [paper-pdf](http://arxiv.org/pdf/2507.22446v1)

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms.

摘要: 尽管深度神经网络取得了广泛成功，但仍然极易受到对抗攻击，从而在安全敏感的应用中构成了重大风险。本文研究了激活函数作为增强模型稳健性的关键但未充分研究的组件。我们提出了Rademacher复杂性降低激活函数（RCR-AF），这是一种新型激活函数，旨在提高概括性和对抗韧性。RCR-AF独特地结合了GELU的优势（包括平滑性、梯度稳定性和负信息保留性）与ReLU理想的单调性，同时通过由两个超参数$\Alpha$和$\gamma$管理的内置剪裁机制控制模型稀疏性和容量。我们基于Rademacher复杂性的理论分析表明，这些参数直接调节模型的Rademacher复杂性，提供了一种增强稳健性的原则方法。全面的实证评估表明，RCR-AF在标准训练下的清晰准确性和对抗训练范式内的对抗鲁棒性方面始终优于广泛使用的替代方案（ReLU、GELU和Swish）。



## **10. Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss**

具有CE损失的对抗性攻击梯度计算相对误差的理论分析 cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22428v1) [paper-pdf](http://arxiv.org/pdf/2507.22428v1)

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy.

摘要: 由于浮点算法引起的梯度计算中的相对误差，使用交叉Entropy（CE）损失的基于对象的对抗攻击经常会被高估。本文对这些错误进行了严格的理论分析，首次对四种不同场景下基于梯度的攻击中的浮点计算错误进行了全面研究：（i）不成功的非目标攻击，（ii）成功的非目标攻击，（iii）不成功的目标攻击，（iv）成功的目标攻击。我们建立了描述不同攻击条件下相对数值误差行为的理论基础，揭示了梯度计算不稳定性中以前未知的模式，并确定浮点下溢和舍入是关键因素。基于这一见解，我们提出了理论MIFPE（T-MIFPE）损失函数，它结合了最佳缩放因子$T = t^*$，以最大限度地减少浮点错误的影响，从而提高对抗性攻击中梯度计算的准确性。对MNIST、CIFAR-10和CIFAR-100数据集的广泛实验表明，T-MIFPE在攻击效力和鲁棒性评估准确性方面优于现有的损失函数，包括CE、C & W、DLR和MIFPE。



## **11. Benchmarking Fraud Detectors on Private Graph Data**

针对私有图表数据对欺诈检测器进行基准测试 cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22347v1) [paper-pdf](http://arxiv.org/pdf/2507.22347v1)

**Authors**: Alexander Goldberg, Giulia Fanti, Nihar Shah, Zhiwei Steven Wu

**Abstract**: We introduce the novel problem of benchmarking fraud detectors on private graph-structured data. Currently, many types of fraud are managed in part by automated detection algorithms that operate over graphs. We consider the scenario where a data holder wishes to outsource development of fraud detectors to third parties (e.g., vendors or researchers). The third parties submit their fraud detectors to the data holder, who evaluates these algorithms on a private dataset and then publicly communicates the results. We propose a realistic privacy attack on this system that allows an adversary to de-anonymize individuals' data based only on the evaluation results. In simulations of a privacy-sensitive benchmark for facial recognition algorithms by the National Institute of Standards and Technology (NIST), our attack achieves near perfect accuracy in identifying whether individuals' data is present in a private dataset, with a True Positive Rate of 0.98 at a False Positive Rate of 0.00. We then study how to benchmark algorithms while satisfying a formal differential privacy (DP) guarantee. We empirically evaluate two classes of solutions: subsample-and-aggregate and DP synthetic graph data. We demonstrate through extensive experiments that current approaches do not provide utility when guaranteeing DP. Our results indicate that the error arising from DP trades off between bias from distorting graph structure and variance from adding random noise. Current methods lie on different points along this bias-variance trade-off, but more complex methods tend to require high-variance noise addition, undermining utility.

摘要: 我们引入了在私人图形结构数据上对欺诈检测器进行基准测试的新颖问题。目前，许多类型的欺诈在一定程度上是通过图形操作的自动检测算法来管理的。我们考虑数据持有者希望将欺诈检测器的开发外包给第三方（例如，供应商或研究人员）。第三方将其欺诈检测器提交给数据持有者，数据持有者在私人数据集上评估这些算法，然后公开传达结果。我们对该系统提出了一种现实的隐私攻击，允许对手仅根据评估结果对个人数据进行去匿名化。在美国国家标准与技术研究院（NIH）对面部识别算法隐私敏感基准的模拟中，我们的攻击在识别个人数据是否存在于私人数据集中方面实现了近乎完美的准确性，真阳性率为0.98，假阳性率为0.00。然后，我们研究如何在满足正式的差异隐私（DP）保证的同时对算法进行基准测试。我们根据经验评估了两类解决方案：子样本和聚合数据和DP合成图数据。我们通过大量实验证明，当前的方法在保证DP时无法提供实用性。我们的结果表明，DP引起的误差在扭曲图结构的偏差和添加随机噪音的方差之间权衡。当前的方法位于这种偏差方差权衡的不同点，但更复杂的方法往往需要添加高方差的噪音，从而削弱了效用。



## **12. Resilient State Recovery using Prior Measurement Support Information**

使用先前测量支持信息进行弹性状态恢复 math.OC

To be published in SIAM Journal on Control and Optimization

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22340v1) [paper-pdf](http://arxiv.org/pdf/2507.22340v1)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Warren E. Dixon

**Abstract**: Resilient state recovery of cyber-physical systems has attracted much research attention due to the unique challenges posed by the tight coupling between communication, computation, and the underlying physics of such systems. By modeling attacks as additive adversary signals to a sparse subset of measurements, this resilient recovery problem can be formulated as an error correction problem. To achieve exact state recovery, most existing results require less than $50\%$ of the measurement nodes to be compromised, which limits the resiliency of the estimators. In this paper, we show that observer resiliency can be further improved by incorporating data-driven prior information. We provide an analytical bridge between the precision of prior information and the resiliency of the estimator. By quantifying the relationship between the estimation error of the weighted $\ell_1$ observer and the precision of the support prior. This quantified relationship provides guidance for the estimator's weight design to achieve optimal resiliency. Several numerical simulations and an application case study are presented to validate the theoretical claims.

摘要: 由于通信、计算和此类系统的基础物理之间的紧密耦合所带来的独特挑战，网络物理系统的弹性状态恢复引起了广泛的研究关注。通过将攻击建模为对稀疏测量子集的添加对手信号，这个弹性恢复问题可以被表述为错误纠正问题。为了实现精确的状态恢复，大多数现有结果需要不到50%的测量节点受到损害，这限制了估计器的弹性。在本文中，我们表明可以通过纳入数据驱动的先验信息进一步提高观察者的弹性。我们在先验信息的精确性和估计器的弹性之间提供了分析桥梁。通过量化加权$\ell_1 $观察者的估计误差与支持先验精度之间的关系。这种量化关系为估计器的权重设计提供指导，以实现最佳弹性。文中给出了几个数值模拟和应用案例研究来验证理论主张。



## **13. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **14. Persistent Backdoor Attacks in Continual Learning**

持续学习中的持续后门攻击 cs.LG

19 pages, 20 figures, 6 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2409.13864v3) [paper-pdf](http://arxiv.org/pdf/2409.13864v3)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.

摘要: 后门攻击对神经网络构成了重大威胁，使对手能够操纵特定输入的模型输出，通常会带来毁灭性的后果，特别是在关键应用中。虽然后门攻击已经在各种背景下进行了研究，但很少有人关注它们在持续学习中的实用性和持久性，特别是在了解随着新数据分布的学习和集成，模型参数的持续更新如何影响这些攻击的有效性方面。为了解决这一差距，我们引入了两种持续的后门攻击-盲任务后门和潜在任务后门-每一种都利用最小的对抗影响。我们的盲任务后门巧妙地改变了损失计算，而不直接控制训练过程，而潜在的任务后门只影响单个任务的训练，所有其他任务的训练都是良性的。我们在各种配置下评估了这些攻击，展示了它们在静态、动态、物理和语义触发下的功效。我们的结果表明，这两种攻击在不同的持续学习算法上始终实现了高成功率，同时有效地规避了SentiNet和I-BAU等最先进的防御。



## **15. Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation**

教我恶作剧：通过知识提炼探索对抗性可转移性 cs.LG

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21992v1) [paper-pdf](http://arxiv.org/pdf/2507.21992v1)

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks.

摘要: 我们研究来自多个异类教师模型的知识提炼（KD）是否可以增强可转移对抗示例的生成。轻量级学生模型使用两种KD策略进行训练：基于课程的切换和联合优化，ResNet 50和DenseNet-161作为教师。然后，经过训练的学生使用FG、FSG和PVD攻击生成对抗性示例，并针对黑匣子目标模型（GoogLeNet）进行评估。我们的结果表明，从多名教师中提取的学生模型的攻击成功率与基于整体的基线相当，同时将对抗性示例生成时间减少六倍。一项消融研究进一步表明，较低的温度设置和硬标签监督的纳入显着增强了可转移性。这些发现表明，KD不仅可以作为一种模型压缩技术，还可以作为提高黑匣子对抗攻击效率和有效性的强大工具。



## **16. ZIUM: Zero-Shot Intent-Aware Adversarial Attack on Unlearned Models**

ZIUM：对未学习模型的零攻击意图感知对抗攻击 cs.CV

Accepted to ICCV2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21985v1) [paper-pdf](http://arxiv.org/pdf/2507.21985v1)

**Authors**: Hyun Jun Yook, Ga San Jhun, Jae Hyun Cho, Min Jeon, Donghyun Kim, Tae Hyung Kim, Youn Kyu Lee

**Abstract**: Machine unlearning (MU) removes specific data points or concepts from deep learning models to enhance privacy and prevent sensitive content generation. Adversarial prompts can exploit unlearned models to generate content containing removed concepts, posing a significant security risk. However, existing adversarial attack methods still face challenges in generating content that aligns with an attacker's intent while incurring high computational costs to identify successful prompts. To address these challenges, we propose ZIUM, a Zero-shot Intent-aware adversarial attack on Unlearned Models, which enables the flexible customization of target attack images to reflect an attacker's intent. Additionally, ZIUM supports zero-shot adversarial attacks without requiring further optimization for previously attacked unlearned concepts. The evaluation across various MU scenarios demonstrated ZIUM's effectiveness in successfully customizing content based on user-intent prompts while achieving a superior attack success rate compared to existing methods. Moreover, its zero-shot adversarial attack significantly reduces the attack time for previously attacked unlearned concepts.

摘要: 机器去学习（MU）从深度学习模型中删除特定数据点或概念，以增强隐私并防止敏感内容生成。对抗性提示可以利用未学习的模型来生成包含已删除概念的内容，从而构成重大的安全风险。然而，现有的对抗性攻击方法在生成符合攻击者意图的内容方面仍然面临挑战，同时识别成功提示需要付出高昂的计算成本。为了应对这些挑战，我们提出了ZIUM，这是一种对Unleared Models的零射击意图感知对抗攻击，它能够灵活定制目标攻击图像以反映攻击者的意图。此外，ZIUM支持零射击对抗攻击，而不需要进一步优化先前攻击的未学习概念。对各种MU场景的评估表明，ZIUM在根据用户意图提示成功定制内容方面的有效性，同时与现有方法相比，其攻击成功率更高。此外，它的零射击对抗攻击大大减少了以前攻击的未学习概念的攻击时间。



## **17. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

任何人都可以越狱：针对LLM和T2 I的预算攻击 cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.

摘要: 尽管在对齐和内容审核方面取得了重大进步，但大型语言模型（LLM）和文本到图像（T2 I）系统仍然容易受到基于预算的攻击（即越狱）。与需要专业知识的传统对抗示例不同，今天的许多越狱都是由日常用户精心设计的，只需措辞巧妙的提示即可。本文对非专家如何通过多回合叙事升级、词汇伪装、隐含链接、虚构模仿和微妙的语义编辑等技术可靠地规避安全机制进行了系统式的调查。我们基于流行API的实证案例研究，提出了跨越文本输出和T2 I模型的预算级越狱策略的统一分类。我们的分析表明，审核管道的每个阶段，从输入过滤到输出验证，都可以通过可访问的策略绕过。最后，我们强调了对上下文感知防御的迫切需要，以反映这些越狱可以在现实世界环境中轻松复制的情况。



## **18. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

没有对抗防御的对抗防御：通过实例级主成分去除增强语言模型稳健性 cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21750v1) [paper-pdf](http://arxiv.org/pdf/2507.21750v1)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.

摘要: 预训练的语言模型（PLM）推动了自然语言处理的重大进展，但仍然容易受到对抗攻击，引发了对其在现实世界应用程序中稳健性的担忧。之前的研究试图通过隐式或显式地在训练过程中引入对抗性扰动来减轻对抗性攻击的影响。虽然这两种策略都增强了稳健性，但它们通常会产生很高的计算成本。在这项工作中，我们提出了一个简单而有效的附加模块，该模块通过删除实例级主成分来增强PLM的对抗鲁棒性，而不依赖于传统的对抗防御或干扰原始训练数据。我们的方法将嵌入空间转换为逼近高斯属性，从而降低其对对抗性扰动的敏感性，同时保留语义关系。这种转换以一种最小化对抗性噪音对决策边界的影响的方式对齐嵌入分布，增强稳健性，而无需对抗性示例或昂贵的训练时间扩展。对八个基准数据集的评估表明，我们的方法提高了对抗稳健性，同时保持了与基线相当的攻击前准确性，实现了稳健性和概括性之间的平衡。



## **19. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

通过潜在对抗训练防御不可预见的失败模式 cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2403.05030v6) [paper-pdf](http://arxiv.org/pdf/2403.05030v6)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，人工智能系统有时会表现出有害的非预期行为。找到和修复这些问题具有挑战性，因为攻击面如此之大--无法彻底搜索可能引发有害行为的输入。红色团队和对抗训练（AT）通常用于提高稳健性，然而，从经验上看，它们很难修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需利用有关漏洞的知识或使用引发漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。在这里，我们使用它来防御没有引发失败模式的例子的失败模式。具体来说，我们使用LAT来删除后门并抵御持续的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常会提高对新型攻击的鲁棒性和干净数据的性能。这表明LAT可以成为一种有前途的工具，用于防御开发人员未明确识别的故障模式。



## **20. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型（LLM）通常会以不受欢迎的方式运行，而这些方式明确进行了微调。例如，LLM红色团队文献产生了各种“越狱”技术，从经过微调的无害模型中引出有害文本。最近关于红色团队、模型编辑和可解释性的工作表明，这一挑战源于（对抗性）微调如何在很大程度上用于抑制而不是消除LLM中不受欢迎的功能。之前的工作引入了潜在对抗训练（LAT），作为提高对广泛类型失败的稳健性的一种方法。这些先前的作品考虑了无针对性的潜在空间攻击，其中对手扰乱潜在激活，以最大限度地增加理想行为示例的损失。无目标LAT可以提供通用类型的稳健性，但不会利用有关特定故障模式的信息。在这里，我们尝试了有针对性的LAT，其中对手试图最大限度地减少特定竞争任务的损失。我们发现它可以增强各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的稳健性，以减少数量级的计算来超越强大的R2 D2基线。其次，我们使用它来在不知道触发器的情况下更有效地删除后门。最后，我们使用它以一种对重新学习更稳健的方式更有效地忘记特定不需要任务的知识。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **21. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

PRism：用于LVLM越狱的具有图像序列操作的程序推理 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.

摘要: 随着大型视觉语言模型（LVLM）的日益复杂，旨在防止有害内容生成的安全对齐机制也取得了进步。然而，这些防御系统仍然容易受到复杂的对抗攻击。现有的越狱方法通常依赖于直接且语义明确的提示，忽略了LVLM如何通过多个推理步骤组成信息的微妙漏洞。本文受到软件安全领域的面向返回编程（opp）技术的启发，提出了一种新颖且有效的越狱框架。我们的方法将有害的指令分解为一系列单独良性的视觉小工具。精心设计的文本提示引导输入序列，促使模型通过其推理过程集成良性视觉小工具，以产生连贯且有害的输出。这使得恶意意图变得紧急，并且难以从任何单个组件中检测到。我们通过对SafeBench和MM-SafetyBench等既定基准进行广泛实验来验证我们的方法，目标是流行的LVLM。结果表明，我们的方法始终且大幅优于最先进模型上的现有基线，实现了近乎完美的攻击成功率（SafeBench上超过0.90），并将ASB提高高达0.39。我们的研究结果揭示了一个关键且未充分探索的漏洞，该漏洞利用了LVLM的合成推理能力，凸显了对保护整个推理过程的防御措施的迫切需求。



## **22. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

我们能结束猫鼠游戏吗？使用LLM和遗传算法模拟自我进化的网络钓鱼攻击 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.

摘要: 预测新出现的攻击方法对于主动网络安全至关重要。大型语言模型（LLM）的最新进展使网络钓鱼消息的自动生成成为可能，并加速了对潜在攻击技术的研究。然而，由于依赖于现有的训练数据，预测未来的威胁仍然具有挑战性。为了解决这一限制，我们提出了一种新的框架，集成了基于LLM的网络钓鱼攻击模拟与遗传算法在心理背景下，使网络钓鱼策略通过与模拟受害者的对抗性交互动态演变。通过使用Llama 3.1的模拟，我们证明了（1）自我进化的网络钓鱼策略采用了越来越复杂的心理操纵技术，超越了天真的LLM生成的攻击，（2）受害者先验知识的变化显着影响攻击策略的演变，（3）不断进化的攻击和适应性防御之间的对抗相互作用创造了猫鼠动态，暴露了网络安全固有的不对称性--攻击者不断完善他们的方法，而防御者则难以全面应对所有不断变化的威胁。我们的方法提供了一种可扩展、具有成本效益的方法来分析网络钓鱼策略和防御的演变，提供了对未来社会工程威胁的见解，并强调了主动网络安全措施的必要性。



## **23. NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples**

NCCR：评估神经网络和对抗示例的鲁棒性 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21483v1) [paper-pdf](http://arxiv.org/pdf/2507.21483v1)

**Authors**: Pu Shi

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust.

摘要: 神经网络最近受到了广泛关注，相关的安全问题也随之而来。许多研究表明，神经网络容易受到经过修改人为干扰的对抗性示例的影响，这些示例太小，无法通过人类感知来区分。人们提出了不同的攻击和防御来解决这些问题，但关于评估神经网络及其输入的稳健性的研究很少。在这项工作中，我们提出了一种名为神经元覆盖变化率（NCCR）的指标来衡量深度学习模型抵抗攻击的能力和对抗性示例的稳定性。当输入受到干扰时，NCCR监控特定选择的神经元输出的变化，并且变化程度较小的网络被认为更稳健。图像识别和说话人识别模型的实验结果表明，我们的指标可以很好地评估神经网络或其输入的鲁棒性。它还可以用于检测输入是否具有对抗性，因为对抗性示例总是不太稳健。



## **24. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

PAR-AdvGAN：通过渐进式自回归AdvGAN提高对抗攻击能力 cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2502.12207v2) [paper-pdf](http://arxiv.org/pdf/2502.12207v2)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR

摘要: 深度神经网络在各个领域都表现出了卓越的性能。然而，它们容易受到对抗性例子的影响，这可能导致错误的预测。生成对抗网络（GAN）可以利用生成器和鉴别器模型快速生成高质量的对抗示例。由于两个模块都以竞争和同步的方式进行训练，因此与传统方法相比，基于GAN的算法（如AdvGAN）可以生成具有更好可移植性的对抗性示例。然而，扰动的产生通常仅限于单次迭代，从而阻止这些示例充分利用方法的潜力。为了解决这个问题，我们引入了一种名为渐进式自动回归AdvGAN（PAR-AdvGAN）的新颖方法。它在渐进生成网络中集成了自回归迭代机制，以制作具有增强攻击能力的对抗性示例。我们通过大规模实验彻底评估了我们的PAR-AdvGAN方法，证明了其优于各种最先进的黑匣子对抗攻击以及原始的AdvGAN的性能。此外，PAR-AdvGAN显着加速了对抗性示例的生成，即Inception-v3模型上的速度高达每秒335.5帧，优于基于梯度的可转移攻击算法。我们的代码可访问：https://github.com/LMBTough/PAR



## **25. Cascading and Proxy Membership Inference Attacks**

级联和代理成员推断攻击 cs.CR

Our code is available at: https://github.com/zealscott/MIA

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21412v1) [paper-pdf](http://arxiv.org/pdf/2507.21412v1)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.

摘要: 成员资格推理攻击（MIA）通过确定数据集中是否包括特定的查询实例来评估经过训练的机器学习模型对其训练数据的揭示程度。我们将现有的MIA分为自适应或非自适应，具体取决于是否允许对手在成员资格查询上训练影子模型。在自适应环境中，对手可以在访问查询实例后训练影子模型，我们强调了利用实例之间成员依赖关系的重要性，并提出了一种名为级联成员推断攻击（CMIA）的攻击不可知框架，该框架通过条件影子训练合并成员依赖关系，以提高成员推断性能。   在非自适应环境中，对手仅限于在获得成员资格查询之前训练影子模型，我们引入代理成员资格推断攻击（PMIA）。PMIA采用代理选择策略，该策略识别与查询实例具有相似行为的样本，并使用其在影子模型中的行为来执行成员资格后验赔率测试以进行成员资格推断。我们对这两种攻击提供了理论分析，大量的实验结果表明，CMIA和PMIA在这两种环境下的表现都大大优于现有的MIA，特别是在低假阳性机制下，这对于评估隐私风险至关重要。



## **26. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

FedStrategist：一个用于联邦学习中自适应和鲁棒聚合的元学习框架 cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14322v2) [paper-pdf](http://arxiv.org/pdf/2507.14322v2)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.

摘要: 联邦学习（FL）为保护隐私的协作人工智能提供了一个范式，但其去中心化性质给建模中毒攻击带来了显着的漏洞。虽然存在许多静态防御，但它们的有效性高度依赖于上下文，通常无法对抗自适应对手或在异类数据环境中。本文介绍了FedStrategist，这是一种新型的元学习框架，它将稳健聚合重新定义为实时、成本感知的控制问题。我们设计了一个轻量级的上下文强盗代理，它基于实时诊断指标从防御库中动态选择最佳聚合规则。通过全面的实验，我们证明没有单一的静态规则是普遍最优的。我们表明，我们的适应性代理能够在不同的场景中成功学习更好的策略，包括“克鲁姆有利”的环境以及针对旨在中和特定诊断信号的复杂“隐形”对手。至关重要的是，我们分析了自相矛盾的场景，即非稳健基线实现了高但受到损害的准确性，并证明我们的代理学习了保守的策略来优先考虑模型完整性。此外，我们证明了代理的策略是可以通过单个“风险容忍度”参数来控制的，允许从业者显式地管理性能和安全性之间的权衡。我们的工作提供了一种新的、实用的、可分析的方法来创建弹性和智能的去中心化人工智能系统。



## **27. Radio Adversarial Attacks on EMG-based Gesture Recognition Networks**

基于EMG的手势识别网络的无线对抗攻击 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21387v1) [paper-pdf](http://arxiv.org/pdf/2507.21387v1)

**Authors**: Hongyi Xie

**Abstract**: Surface electromyography (EMG) enables non-invasive human-computer interaction in rehabilitation, prosthetics, and virtual reality. While deep learning models achieve over 97% classification accuracy, their vulnerability to adversarial attacks remains largely unexplored in the physical domain. We present ERa Attack, the first radio frequency (RF) adversarial method targeting EMG devices through intentional electromagnetic interference (IEMI). Using low-power software-defined radio transmitters, attackers inject optimized RF perturbations to mislead downstream models. Our approach bridges digital and physical domains: we generate adversarial perturbations using Projected Gradient Descent, extract 50-150 Hz components via inverse STFT, and employ synchronization-free strategies (constant spectrum noise or narrowband modulation). Perturbations, constrained to 1-10% of signal amplitude, are amplitude-modulated onto 433 MHz carriers. Experiments on the Myo Dataset (7 gestures, 350 samples) demonstrate significant impact: at 1 meter and 0 dBm transmission power, classification accuracy drops from 97.8% to 58.3%, with 41.7% misclassification rate and 25.6% targeted attack success rate. Attack effectiveness decreases exponentially with distance, recovering to 85% accuracy at 3 meters. Increasing power to 10 dBm reduces accuracy by an additional 15% at 1 meter. This work pioneers RF-based adversarial attacks on EMG recognition systems, revealing critical vulnerabilities in safety-critical applications. We quantify attack effectiveness across different perturbation modes and distances, and propose defenses including hardware shielding, spectrum monitoring, and adversarial training. Our findings inform the design of robust EMG systems against electromagnetic threats.

摘要: 表面肌电信号（EMG）实现康复、假肢和虚拟现实中的非侵入性人机交互。虽然深度学习模型的分类准确率超过97%，但它们对对抗攻击的脆弱性在物理领域基本上尚未被探索。我们介绍了ERa Attack，这是第一种通过故意电磁干扰（IEMI）针对EMG设备的射频（RF）对抗方法。使用低功耗软件定义的无线电发射机，攻击者注入优化的RF扰动来误导下游模型。我们的方法架起数字和物理领域的桥梁：我们使用投影梯度下降来生成对抗性扰动，通过逆STFT提取50-150 Hz分量，并采用无同步策略（恒定频谱噪音或窄频调制）。限制在信号幅度的1-10%的扰动被幅度调制到433 MHz载体上。Myo数据集（7个手势，350个样本）的实验显示了显着的影响：在1米和0 dBm传输功率下，分类准确率从97.8%下降到58.3%，误分类率为41.7%，目标攻击成功率为25.6%。攻击有效性随着距离的增加呈指数级下降，在3米处恢复到85%的准确率。将功率增加到10分贝会使1米处的准确性额外降低15%。这项工作开创了对EMG识别系统的基于RF的对抗攻击，揭示了安全关键应用程序中的关键漏洞。我们量化了不同扰动模式和距离的攻击有效性，并提出了包括硬件屏蔽、频谱监控和对抗训练在内的防御措施。我们的研究结果为针对电磁威胁的稳健EMG系统的设计提供了信息。



## **28. On Post-Quantum Cryptography Authentication for Quantum Key Distribution**

量子密钥分发的后量子密码认证 quant-ph

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21325v1) [paper-pdf](http://arxiv.org/pdf/2507.21325v1)

**Authors**: Juan Antonio Vieira Giestinhas, Timothy Spiller

**Abstract**: The traditional way for a Quantum Key Distribution (QKD) user to join a quantum network is by authenticating themselves using pre-shared key material. While this approach is sufficient for small-scale networks, it becomes impractical as the network grows, due to the total quadratic increase in the number of pre-shared keys required. To address this scalability issue, Public Key Infrastructure (PKI) combined with Post-Quantum Cryptography (PQC) offers a more scalable solution, allowing users to authenticate the QKD traffic remotely to obtain information-theoretical secure (ITS) keys under the presented assumptions. Unlike traditional PKI, which relies on classical cryptographic algorithms such as RSA, the approach presented in this paper leverages PQC algorithms that are believed to be resistant to quantum attacks. Similarly to the SIGMA or TLS protocols, authentication, confidentiality, and integrity are achievable against bounded adversaries to ensure secure and scalable quantum networks.

摘要: 量子密钥分发（QKD）用户加入量子网络的传统方法是使用预共享密钥材料对自己进行身份验证。虽然这种方法对于小规模网络来说已经足够了，但随着网络的增长，它变得不切实际，因为所需的预共享密钥数量的总二次增加。为了解决这个可扩展性问题，公钥基础设施（公钥基础设施）与后量子密码学（PQC）相结合提供了一种更具可扩展性的解决方案，允许用户远程验证QKD流量，以在所提出的假设下获得信息理论安全（ITS）密钥。与依赖RSA等经典加密算法的传统公钥不同，本文中提出的方法利用了被认为可以抵抗量子攻击的PQC算法。与SIGMA或SSL协议类似，针对有界对手可以实现身份验证、机密性和完整性，以确保量子网络的安全和可扩展性。



## **29. Adversarial attacks and defenses in explainable artificial intelligence: A survey**

可解释人工智能中的对抗性攻击和防御：一项调查 cs.CR

Accepted by Information Fusion

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2306.06123v4) [paper-pdf](http://arxiv.org/pdf/2306.06123v4)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning (AdvML) highlight the limitations and vulnerabilities of state-of-the-art explanation methods, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This survey provides a comprehensive overview of research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We introduce a unified notation and taxonomy of methods facilitating a common ground for researchers and practitioners from the intersecting research fields of AdvML and XAI. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI). Future work should address improving explanation methods and evaluation protocols to take into account the reported safety issues.

摘要: 可解释人工智能（XAI）方法被描述为调试和信任统计和深度学习模型以及解释其预测的补救措施。然而，对抗性机器学习（AdvML）的最新进展凸显了最先进解释方法的局限性和漏洞，使其安全性和可信性受到质疑。当应用于高风险决策和知识发现时，操纵、愚弄或粉饰模型推理证据的可能性会产生有害的后果。这项调查全面概述了有关对机器学习模型解释的对抗性攻击以及公平性指标的研究。我们引入了统一的方法符号和分类法，为AdvML和XAI交叉研究领域的研究人员和从业者建立共同点。我们讨论如何防御攻击和设计稳健的解释方法。我们列出了XAI中现有的不安全感，并概述了对抗性XAI（AdvXAI）中新兴的研究方向。未来的工作应解决改进解释方法和评估方案，以考虑报告的安全问题。



## **30. Improving Adversarial Robustness Through Adaptive Learning-Driven Multi-Teacher Knowledge Distillation**

通过自适应学习驱动的多教师知识提炼提高对抗稳健性 cs.CV

11 pages

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20996v1) [paper-pdf](http://arxiv.org/pdf/2507.20996v1)

**Authors**: Hayat Ullah, Syed Muhammad Talha Zaidi, Arslan Munir

**Abstract**: Convolutional neural networks (CNNs) excel in computer vision but are susceptible to adversarial attacks, crafted perturbations designed to mislead predictions. Despite advances in adversarial training, a gap persists between model accuracy and robustness. To mitigate this issue, in this paper, we present a multi-teacher adversarial robustness distillation using an adaptive learning strategy. Specifically, our proposed method first trained multiple clones of a baseline CNN model using an adversarial training strategy on a pool of perturbed data acquired through different adversarial attacks. Once trained, these adversarially trained models are used as teacher models to supervise the learning of a student model on clean data using multi-teacher knowledge distillation. To ensure an effective robustness distillation, we design an adaptive learning strategy that controls the knowledge contribution of each model by assigning weights as per their prediction precision. Distilling knowledge from adversarially pre-trained teacher models not only enhances the learning capabilities of the student model but also empowers it with the capacity to withstand different adversarial attacks, despite having no exposure to adversarial data. To verify our claims, we extensively evaluated our proposed method on MNIST-Digits and Fashion-MNIST datasets across diverse experimental settings. The obtained results exhibit the efficacy of our multi-teacher adversarial distillation and adaptive learning strategy, enhancing CNNs' adversarial robustness against various adversarial attacks.

摘要: 卷积神经网络（CNN）在计算机视觉方面表现出色，但很容易受到对抗性攻击和旨在误导预测的精心设计的干扰。尽管对抗训练取得了进步，但模型准确性和稳健性之间仍然存在差距。为了缓解这个问题，在本文中，我们使用自适应学习策略提出了一种多教师对抗鲁棒性提炼。具体来说，我们提出的方法首先在通过不同对抗攻击获取的受干扰数据池上使用对抗训练策略训练基线CNN模型的多个克隆。训练后，这些经过对抗训练的模型将用作教师模型，使用多教师知识蒸馏来监督学生模型在干净数据上的学习。为了确保有效的鲁棒性提炼，我们设计了一种自适应学习策略，通过根据每个模型的预测精度分配权重来控制每个模型的知识贡献。从经过对抗预训练的教师模型中提取知识不仅可以增强学生模型的学习能力，而且还使其具有抵御不同对抗攻击的能力，尽管没有接触到对抗数据。为了验证我们的说法，我们在不同实验环境中对MNIST-Digits和Fashion-MNIST数据集进行了广泛评估。所获得的结果展示了我们的多教师对抗提炼和自适应学习策略的功效，增强了CNN针对各种对抗攻击的对抗鲁棒性。



## **31. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

运输网络物理系统支持的大语言模型威胁建模框架 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2506.00831v2) [paper-pdf](http://arxiv.org/pdf/2506.00831v2)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Existing threat modeling frameworks related to transportation cyber-physical systems (CPS) are often narrow in scope, labor-intensive, and require substantial cybersecurity expertise. To this end, we introduce the Transportation Cybersecurity and Resiliency Threat Modeling Framework (TraCR-TMF), a large language model (LLM)-based threat modeling framework for transportation CPS that requires limited cybersecurity expert intervention. TraCR-TMF identifies threats, potential attack techniques, and relevant countermeasures for transportation CPS. Three LLM-based approaches support these identifications: (i) a retrieval-augmented generation approach requiring no cybersecurity expert intervention, (ii) an in-context learning approach with low expert intervention, and (iii) a supervised fine-tuning approach with moderate expert intervention. TraCR-TMF offers LLM-based attack path identification for critical assets based on vulnerabilities across transportation CPS entities. Additionally, it incorporates the Common Vulnerability Scoring System (CVSS) scores of known exploited vulnerabilities to prioritize threat mitigations. The framework was evaluated through two cases. First, the framework identified relevant attack techniques for various transportation CPS applications, 73% of which were validated by cybersecurity experts as correct. Second, the framework was used to identify attack paths for a target asset in a real-world cyberattack incident. TraCR-TMF successfully predicted exploitations, like lateral movement of adversaries, data exfiltration, and data encryption for ransomware, as reported in the incident. These findings show the efficacy of TraCR-TMF in transportation CPS threat modeling, while reducing the need for extensive involvement of cybersecurity experts. To facilitate real-world adoptions, all our codes are shared via an open-source repository.

摘要: 与交通网络物理系统（CPS）相关的现有威胁建模框架通常范围狭窄、劳动密集型，并且需要大量的网络安全专业知识。为此，我们引入了交通网络安全和弹性威胁建模框架（TraCR-SYS），这是一个基于大型语言模型（LLM）的交通CPS威胁建模框架，需要有限的网络安全专家干预。TraCR-SYS识别交通CPS的威胁、潜在攻击技术和相关对策。三种基于LLM的方法支持这些识别：（i）不需要网络安全专家干预的检索增强生成方法，（ii）具有低专家干预的上下文学习方法，（iii）具有适度专家干预的监督微调方法。TraCR-SYS根据运输CPS实体之间的漏洞为关键资产提供基于LLM的攻击路径识别。此外，它还结合了已知被利用漏洞的通用漏洞评分系统（CVD）评分，以确定威胁缓解的优先顺序。该框架通过两个案例进行了评估。首先，该框架确定了各种交通CPS应用程序的相关攻击技术，其中73%经网络安全专家验证为正确。其次，该框架用于识别现实世界网络攻击事件中目标资产的攻击路径。正如该事件中所报道的那样，TraCR-SYS成功预测了攻击行为，例如对手的横向移动、数据泄露和勒索软件的数据加密。这些发现表明了TraCR-SYS在交通CPS威胁建模中的功效，同时减少了网络安全专家广泛参与的需要。为了促进现实世界的采用，我们所有的代码都通过开源存储库共享。



## **32. Enhancing generalization in high energy physics using white-box adversarial attacks**

使用白盒对抗攻击增强高能物理学的概括性 hep-ph

14 pages, 7 figures, 10 tables, 3 algorithms, published in Physical  Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2411.09296v3) [paper-pdf](http://arxiv.org/pdf/2411.09296v3)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight-space attacks and feature-space attacks. To study and quantify the sharpness of different local minima, this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.

摘要: 机器学习在粒子物理学背景下变得越来越受欢迎。使用标记蒙特卡洛（MC）模拟的监督学习仍然是标准模型之外最广泛使用的区分信号的方法之一。然而，本文表明，监督模型可能过度依赖蒙特卡洛模拟的伪影和逼近，这可能会限制它们很好地概括为真实数据的能力。本研究旨在通过降低局部极小值的清晰度来增强监督模型的概括性。它回顾了四种不同的白盒对抗攻击在对希格斯玻色子衰变信号进行分类的背景下的应用。攻击分为重量空间攻击和特征空间攻击。为了研究和量化不同局部极小值的清晰度，本文提出了两种分析方法：梯度上升和简化海森特征值分析。结果表明，白盒对抗攻击显着提高了概括性能，尽管计算复杂性有所增加。



## **33. Next-Generation Quantum Neural Networks: Enhancing Efficiency, Security, and Privacy**

下一代量子神经网络：提高效率、安全性和隐私 quant-ph

4 pages, 6 figures. Accepted at IOLTS 2025

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20537v1) [paper-pdf](http://arxiv.org/pdf/2507.20537v1)

**Authors**: Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Mohamed Bennai, Muhammad Shafique

**Abstract**: This paper provides an integrated perspective on addressing key challenges in developing reliable and secure Quantum Neural Networks (QNNs) in the Noisy Intermediate-Scale Quantum (NISQ) era. In this paper, we present an integrated framework that leverages and combines existing approaches to enhance QNN efficiency, security, and privacy. Specifically, established optimization strategies, including efficient parameter initialization, residual quantum circuit connections, and systematic quantum architecture exploration, are integrated to mitigate issues such as barren plateaus and error propagation. Moreover, the methodology incorporates current defensive mechanisms against adversarial attacks. Finally, Quantum Federated Learning (QFL) is adopted within this framework to facilitate privacy-preserving collaborative training across distributed quantum systems. Collectively, this synthesized approach seeks to enhance the robustness and real-world applicability of QNNs, laying the foundation for reliable quantum-enhanced machine learning applications in finance, healthcare, and cybersecurity.

摘要: 本文提供了一个综合的视角，探讨了如何应对在有噪音的中间规模量子（NISQ）时代开发可靠和安全的量子神经网络（QNN）的关键挑战。在本文中，我们提出了一个集成框架，该框架利用并结合现有方法来提高QNN的效率、安全性和隐私性。具体来说，集成了已建立的优化策略，包括高效的参数初始化、剩余量子电路连接和系统性量子架构探索，以缓解贫瘠平台和错误传播等问题。此外，该方法结合了当前针对对抗性攻击的防御机制。最后，在该框架内采用量子联合学习（QFL），以促进跨分布式量子系统的隐私保护协作训练。总的来说，这种综合方法旨在增强QNN的稳健性和现实世界的适用性，为金融、医疗保健和网络安全领域可靠的量子增强机器学习应用奠定基础。



## **34. A Hybrid Classical-Quantum Rainbow Table Attack on Human Passwords**

对人类密码的混合经典量子彩虹表攻击 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14600v2) [paper-pdf](http://arxiv.org/pdf/2507.14600v2)

**Authors**: MA. Khajeian

**Abstract**: Long, human-generated passwords pose significant challenges to both classical and quantum attacks due to their irregular structure and large search space. In this work, we propose an enhanced classical-quantum hybrid attack specifically designed for this scenario. Our approach constructs rainbow tables using dictionary-based password generation augmented with transformation rules that better capture real-world user behavior. These tables are organized into buckets, enabling faster lookup and reduced space complexity. For the search within each bucket, we employ a distributed exact variant of Grover's algorithm. This method provides deterministic success and significantly lower circuit depth, enhancing robustness against noise-particularly depolarizing errors common in near-term quantum devices. Overall, our hybrid framework improves the efficiency and practicality of password recovery for long, human-readable passwords in realistic adversarial settings.

摘要: 人类生成的长密码由于其结构不规则和搜索空间大，对经典和量子攻击构成了重大挑战。在这项工作中，我们提出了一种专门为这种场景设计的增强型经典量子混合攻击。我们的方法使用基于字典的密码生成来构建彩虹表，该密码生成添加了转换规则，可以更好地捕捉现实世界的用户行为。这些表被组织到桶中，从而实现更快的查找并降低空间复杂性。对于每个桶内的搜索，我们采用Grover算法的分布式精确变体。这种方法提供了确定性的成功和显着降低的电路深度，增强了对噪音的鲁棒性，特别是近期量子设备中常见的去极化误差。总体而言，我们的混合框架提高了在现实对抗环境中对人类可读的长密码恢复的效率和实用性。



## **35. Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards**

意外漏洞：微调中改变模型保障措施的因素 cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2505.16789v2) [paper-pdf](http://arxiv.org/pdf/2505.16789v2)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.

摘要: 随着大型语言模型（LLM）的普及，它们对对抗攻击的脆弱性成为主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会无意中在基础模型中引入漏洞。在这项工作中，我们调查了意外漏洞，即由微调数据特征引起的意外漏洞。我们首先识别多个实验数据集中的潜在相关因素，例如语言特征、语义相似性和毒性。然后，我们评估这些微调模型的对抗稳健性，分析角色转变和可解释性特征，以了解数据集因素如何影响攻击成功率。最后，我们探索了因果关系，为对抗性防御策略提供了新的见解，强调了数据集设计在保留模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_vulnerability上获取。



## **36. Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition**

人工智能代理部署中的安全挑战：大规模公开竞争的见解 cs.AI

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20526v1) [paper-pdf](http://arxiv.org/pdf/2507.20526v1)

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment.

摘要: 最近的进展使LLM驱动的AI代理能够通过将语言模型推理与工具，内存和Web访问相结合来自主执行复杂的任务。但是，这些系统在现实环境中，特别是在受到攻击的情况下，是否能够遵循部署策略呢？为了进行调查，我们进行了迄今为止最大规模的公开红队竞赛，目标是44个现实部署场景中的22个前沿人工智能代理。参与者提交了180万次注入攻击，其中超过60，000次成功引发了违反政策的行为，如未经授权的数据访问，非法金融行为和不遵守监管规定。我们使用这些结果来构建Agent Red Teaming（ART）基准（一组精心策划的高影响力攻击），并在19个最先进的模型上对其进行评估。几乎所有代理在10-100个查询内的大多数行为都会违反策略，并且攻击跨模型和任务的可转移性很高。重要的是，我们发现代理稳健性与模型大小、能力或推断时计算之间的相关性有限，这表明需要针对对抗性滥用采取额外的防御措施。我们的研究结果凸显了当今人工智能代理中的关键且持续存在的漏洞。通过发布ART基准和附带的评估框架，我们的目标是支持更严格的安全评估并推动更安全的代理部署取得进展。



## **37. When and Where do Data Poisons Attack Textual Inversion?**

数据毒药何时何地攻击文本倒置？ cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.10578v3) [paper-pdf](http://arxiv.org/pdf/2507.10578v3)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10

摘要: 中毒攻击对扩散模型（DM）的鲁棒性构成了重大挑战。本文系统地分析了中毒攻击文本倒置（TI）的时间和地点，文本倒置（TI）是一种广泛使用的DM个性化技术。我们首先介绍语义敏感度地图，这是一种用于可视化中毒对文本嵌入影响的新颖方法。其次，我们识别并通过实验验证DM在跨时间步上表现出非均匀的学习行为，重点关注低噪音样本。中毒攻击继承了这种偏见，并主要在较低的时间步注入对抗信号。最后，我们观察到对抗信号分散了学习对训练数据中相关概念区域的注意力，从而破坏了TI过程。基于这些见解，我们提出了安全区训练（SZT），这是一种由3个关键组件组成的新型防御机制：（1）JPEG压缩以削弱高频毒物信号，（2）TI训练期间限制高时步，以避免较低时步的对抗信号，（3）损失掩蔽以将学习限制在相关区域。多种中毒方法的广泛实验表明，SZT极大地增强了TI针对所有中毒攻击的稳健性，提高了生成质量，超出了之前发布的防御措施。代码：www.github.com/JStyborski/Diff_Lab数据：www.github.com/JStyborski/NC10



## **38. EdgeAgentX-DT: Integrating Digital Twins and Generative AI for Resilient Edge Intelligence in Tactical Networks**

EdgeAgentX-DT：集成数字双胞胎和生成人工智能，在战术网络中实现弹性边缘情报 cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21196v1) [paper-pdf](http://arxiv.org/pdf/2507.21196v1)

**Authors**: Abir Ray

**Abstract**: We introduce EdgeAgentX-DT, an advanced extension of the EdgeAgentX framework that integrates digital twin simulations and generative AI-driven scenario training to significantly enhance edge intelligence in military networks. EdgeAgentX-DT utilizes network digital twins, virtual replicas synchronized with real-world edge devices, to provide a secure, realistic environment for training and validation. Leveraging generative AI methods, such as diffusion models and transformers, the system creates diverse and adversarial scenarios for robust simulation-based agent training. Our multi-layer architecture includes: (1) on-device edge intelligence; (2) digital twin synchronization; and (3) generative scenario training. Experimental simulations demonstrate notable improvements over EdgeAgentX, including faster learning convergence, higher network throughput, reduced latency, and improved resilience against jamming and node failures. A case study involving a complex tactical scenario with simultaneous jamming attacks, agent failures, and increased network loads illustrates how EdgeAgentX-DT sustains operational performance, whereas baseline methods fail. These results highlight the potential of digital-twin-enabled generative training to strengthen edge AI deployments in contested environments.

摘要: 我们引入了EdgeAgentX-DT，这是EdgeAgentX框架的高级扩展，集成了数字双胞胎模拟和生成式人工智能驱动场景训练，以显着增强军事网络中的边缘智能。EdgeAgentX-DT利用网络数字双胞胎、与现实世界边缘设备同步的虚拟副本，为培训和验证提供安全、真实的环境。该系统利用生成式人工智能方法，例如扩散模型和变形器，为稳健的基于模拟的代理训练创建多样化且对抗的场景。我们的多层架构包括：（1）设备上边缘智能;（2）数字双胞胎同步;（3）生成式场景训练。实验模拟表明，与EdgeAgentX相比，有显着改进，包括更快的学习收敛、更高的网络吞吐量、减少的延迟以及更好的抗干扰和节点故障的弹性。一个涉及复杂战术场景的案例研究，同时存在干扰攻击、代理故障和网络负载增加，说明了EdgeAgentX-DT如何维持运营性能，而基线方法却失败。这些结果凸显了数字双胞胎生成式训练在有争议环境中加强边缘人工智能部署的潜力。



## **39. Is Crunching Public Data the Right Approach to Detect BGP Hijacks?**

破解公共数据是检测BNP劫持的正确方法吗？ cs.CR

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.20434v1) [paper-pdf](http://arxiv.org/pdf/2507.20434v1)

**Authors**: Alessandro Giaconia, Muoi Tran, Laurent Vanbever, Stefano Vissicchio

**Abstract**: The Border Gateway Protocol (BGP) remains a fragile pillar of Internet routing. BGP hijacks still occurr daily. While full deployment of Route Origin Validation (ROV) is ongoing, attackers have already adapted, launching post-ROV attacks such as forged-origin hijacks. To detect these, recent approaches like DFOH [Holterbach et al., USENIX NSDI '24] and BEAM [Chen et al., USENIX Security '24] apply machine learning (ML) to analyze data from globally distributed BGP monitors, assuming anomalies will stand out against historical patterns. However, this assumption overlooks a key threat: BGP monitors themselves can be misled by adversaries injecting bogus routes. This paper shows that state-of-the-art hijack detection systems like DFOH and BEAM are vulnerable to data poisoning. Using large-scale BGP simulations, we show that attackers can evade detection with just a handful of crafted announcements beyond the actual hijack. These announcements are indeed sufficient to corrupt the knowledge base used by ML-based defenses and distort the metrics they rely on. Our results highlight a worrying weakness of relying solely on public BGP data.

摘要: 边界网关协议（BNP）仍然是互联网路由的脆弱支柱。BNP劫持事件仍然每天发生。虽然路由起源验证（RST）的全面部署正在进行中，但攻击者已经进行了调整，发起了伪起源劫持等RST后攻击。为了检测这些，最近的方法，例如DFOH [Holterbach等人，USENIX NSDI ' 24]和BEAM [Chen等人，USENIX Security ' 24]应用机器学习（ML）来分析来自全球分布的EDI监视器的数据，假设异常将与历史模式相比较。然而，这一假设忽视了一个关键威胁：BNP监控器本身可能会被注入虚假路由的对手误导。本文表明，DFOH和BEAM等最先进的劫持检测系统很容易受到数据中毒的影响。使用大规模的EDI模拟，我们表明，攻击者只需在实际劫持之外发布少量精心设计的公告就可以逃避检测。这些公告确实足以破坏基于ML的防御系统使用的知识库，并扭曲他们所依赖的指标。我们的结果凸显了仅依赖公共EDI数据的令人担忧的弱点。



## **40. Real-time Factuality Assessment from Adversarial Feedback**

来自对抗反馈的实时事实评估 cs.CL

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2410.14651v3) [paper-pdf](http://arxiv.org/pdf/2410.14651v3)

**Authors**: Sanxing Chen, Yukun Huang, Bhuwan Dhingra

**Abstract**: We show that existing evaluations for assessing the factuality of news from conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors-even after their knowledge cutoffs. This suggests that recent popular false information from such sources can be easily identified due to its likely presence in pre-training/retrieval corpora or the emergence of salient, yet shallow, patterns in these datasets. Instead, we argue that a proper factuality evaluation dataset should test a model's ability to reason about current events by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive variants that challenge LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both evaluating and generating challenging news examples, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG-based evaluation helps discover more deceitful patterns.

摘要: 我们表明，现有的评估来自传统来源的新闻真实性的评估（例如事实核查网站上的声明）会随着时间的推移而导致基于LLM的检测器具有很高的准确性--即使在他们的知识截止之后。这表明，来自此类来源的最近流行的虚假信息可以很容易地识别，因为它可能存在于预训练/检索库中，或者这些数据集中出现了显着但浅的模式。相反，我们认为，适当的真实性评估数据集应该测试模型通过检索和阅读相关证据来推理当前事件的能力。为此，我们开发了一种新颖的管道，该管道利用来自基于RAG的检测器的自然语言反馈来迭代地将实时新闻修改为挑战LLM的欺骗性变体。对于强大的基于RAG的GPT-4 o检测器，我们的迭代重写将二元分类ROC-UC绝对减少了17.5%。我们的实验揭示了RAG在评估和生成具有挑战性的新闻示例方面的重要作用，因为免检索的LLM检测器容易受到不可见事件和对抗性攻击的影响，而基于RAG的评估的反馈有助于发现更多欺骗性模式。



## **41. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

16 pages, 5 figures

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2504.14348v4) [paper-pdf](http://arxiv.org/pdf/2504.14348v4)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this paper, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach incorporates two key coordinated components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to construct the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms state-of-the-art attacks, achieving at least a +30.1% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在本文中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法包含两个关键的协调组成部分。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来构建黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于最先进的攻击，在不同任务中的攻击成功率至少提高+30.1%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **42. The DeepSpeak Dataset**

DeepSpeak数据集 cs.CV

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2408.05366v4) [paper-pdf](http://arxiv.org/pdf/2408.05366v4)

**Authors**: Sarah Barrington, Matyas Bohacek, Hany Farid

**Abstract**: Deepfakes represent a growing concern across domains such as impostor hiring, fraud, and disinformation. Despite significant efforts to develop robust detection classifiers to distinguish the real from the fake, commonly used training datasets remain inadequate: relying on low-quality and outdated deepfake generators, consisting of content scraped from online repositories without participant consent, lacking in multimodal coverage, and rarely employing identity-matching protocols to ensure realistic fakes. To overcome these limitations, we present the DeepSpeak dataset, a diverse and multimodal dataset comprising over 100 hours of authentic and deepfake audiovisual content. We contribute: i) more than 50 hours of real, self-recorded data collected from 500 diverse and consenting participants using a custom-built data collection tool, ii) more than 50 hours of state-of-the-art audio and visual deepfakes generated using 14 video synthesis engines and three voice cloning engines, and iii) an embedding-based, identity-matching approach to ensure the creation of convincing, high-quality identity swaps that realistically simulate adversarial deepfake attacks. We also perform large-scale evaluations of state-of-the-art deepfake detectors and show that, without retraining, these detectors fail to generalize to the DeepSpeak dataset. These evaluations highlight the importance of a large and diverse dataset containing deepfakes from the latest generative-AI tools.

摘要: Deepfakes在冒名顶替招聘、欺诈和虚假信息等领域引起了越来越大的关注。尽管人们付出了巨大的努力来开发稳健的检测分类器来区分真假，但常用的训练数据集仍然不足：依赖低质量且过时的深度伪造生成器，由未经参与者同意从在线存储库中抓取的内容组成，缺乏多模式覆盖，并且很少使用身份匹配协议来确保真实的假货。为了克服这些限制，我们提供了DeepSpeak数据集，这是一个多元化的多模式数据集，包括超过100小时的真实和深度伪造视听内容。我们贡献：i）使用定制的数据收集工具从500名不同和同意的参与者收集的超过50小时的真实，自我记录的数据，ii）使用14个视频合成引擎和3个语音克隆引擎生成的超过50小时的最先进的音频和视觉深度伪造，以及iii）基于嵌入的身份匹配方法，以确保创建令人信服的，高质量的身份交换，逼真地模拟对抗性深度伪造攻击。我们还对最先进的deepfake检测器进行了大规模评估，并表明，如果没有重新训练，这些检测器无法推广到DeepSpeak数据集。这些评估强调了包含来自最新生成AI工具的deepfake的大型且多样化数据集的重要性。



## **43. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

迈向更稳健的检索增强生成：在对抗性中毒攻击下评估RAG cs.IR

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2412.16708v2) [paper-pdf](http://arxiv.org/pdf/2412.16708v2)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into the retrieval corpus can mislead models into producing factually incorrect outputs. In this paper, we present a rigorously controlled empirical study of how RAG systems behave under such attacks and how their robustness can be improved. On the generation side, we introduce a structured taxonomy of context types-adversarial, untouched, and guiding-and systematically analyze their individual and combined effects on model outputs. On the retrieval side, we evaluate several retrievers to measure how easily they expose LLMs to adversarial contexts. Our findings also reveal that "skeptical prompting" can activate LLMs' internal reasoning, enabling partial self-defense against adversarial passages, though its effectiveness depends strongly on the model's reasoning capacity. Together, our experiments (code available at https://github.com/JinyanSu1/eval_PoisonRaG) and analysis provide actionable insights for designing safer and more resilient RAG systems, paving the way for more reliable real-world deployments.

摘要: 检索增强生成（RAG）系统已成为减轻LLM幻觉并增强其在知识密集型领域的性能的有希望的解决方案。然而，这些系统很容易受到对抗性中毒攻击，其中注入检索库的恶意段落可能会误导模型产生事实上不正确的输出。在本文中，我们对RAG系统在此类攻击下如何表现以及如何提高其鲁棒性进行了严格控制的实证研究。在生成方面，我们引入了上下文类型的结构化分类法（对抗性、未受影响和引导性），并系统地分析它们对模型输出的单独和组合影响。在检索方面，我们评估了几个检索器，以衡量它们将LLM暴露于对抗性上下文的容易程度。我们的研究结果还表明，“怀疑提示”可以激活LLM的内部推理，使部分自我防御对抗性通道，虽然它的有效性在很大程度上取决于模型的推理能力。总之，我们的实验（代码可在https：//github.com/JinyanSu1/eval_PoisonRaG获得）和分析为设计更安全、更有弹性的RAG系统提供了可操作的见解，为更可靠的现实部署铺平了道路。



## **44. BadPatch: Diffusion-Based Generation of Physical Adversarial Patches**

Badpatch：基于扩散的物理对抗补丁的生成 cs.CV

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2412.01440v4) [paper-pdf](http://arxiv.org/pdf/2412.01440v4)

**Authors**: Zhixiang Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose BadPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using BadPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.

摘要: 印刷在衣服上的物理对抗补丁可以使个人能够躲避人体检测器，但大多数现有方法都优先考虑攻击有效性而不是隐蔽性，从而导致美观的补丁。虽然生成式对抗网络和扩散模型可以产生看起来更自然的补丁，但它们往往无法平衡隐蔽性与攻击有效性，并且缺乏用户定制的灵活性。为了解决这些限制，我们提出了Badpatch，这是一种新型的基于扩散的框架，用于生成可定制的和自然主义的对抗补丁。我们的方法允许用户从参考图像（而不是随机噪音）开始，并结合面罩来创建各种形状的补丁，而不仅仅限于正方形。为了在扩散过程中保留原始语义，我们采用零文本倒置将随机噪音样本映射到单个输入图像，并通过不完全扩散优化（IDO）生成补丁。我们的方法在保持自然外观的同时实现了与最先进的非自然主义补丁相当的攻击性能。使用Badpatch，我们构建了AdvT-shirt-1 K，这是第一个物理对抗性T恤数据集，包含在不同场景中捕获的一千多张图像。AdvT-shirt-1 K可以作为训练或测试未来防御方法的有用数据集。



## **45. Authenticated Sublinear Quantum Private Information Retrieval**

认证的亚线性量子私有信息检索 quant-ph

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2504.04041v3) [paper-pdf](http://arxiv.org/pdf/2504.04041v3)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.

摘要: 利用量子相对熵和互信息给出了一个新的通信复杂度下界，改进了以往经典的基于熵的结果。通过利用Uhlmann引理和量子Pinsker不等式，作者为信息理论安全性建立了更严格的界限，证明了量子协议在平衡隐私和效率方面天生优于经典协议。还探讨了对称量子私有信息检索（QPIR）协议，该协议实现了次线性通信复杂性，同时确保了对似是而非的对手的鲁棒性：一个基于后量子密码学的协议，可以为似是而非的服务器进行身份验证;一个基于环LWE的协议，用于单服务器设置中的后量子安全，确保了对量子攻击的鲁棒性;针对硬件实用性进行了优化的多服务器协议，在保持次线性效率的同时减少了实现开销。这些协议解决了安全数据库查询中的关键漏洞，提供了比经典线性复杂性方法呈指数级的通信改进。该工作还分析了量子似是而非的对手下的安全权衡，为隐私和正确性提供理论保证。



## **46. Trivial Trojans: How Minimal MCP Servers Enable Cross-Tool Exfiltration of Sensitive Data**

琐碎的特洛伊木马：最小的LCP服务器如何实现敏感数据的跨工具泄露 cs.CR

Abstract submitted to the Technical AI Governance Forum 2025  (https://www.techgov.ai/)

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.19880v1) [paper-pdf](http://arxiv.org/pdf/2507.19880v1)

**Authors**: Nicola Croce, Tobin South

**Abstract**: The Model Context Protocol (MCP) represents a significant advancement in AI-tool integration, enabling seamless communication between AI agents and external services. However, this connectivity introduces novel attack vectors that remain largely unexplored. This paper demonstrates how unsophisticated threat actors, requiring only basic programming skills and free web tools, can exploit MCP's trust model to exfiltrate sensitive financial data. We present a proof-of-concept attack where a malicious weather MCP server, disguised as benign functionality, discovers and exploits legitimate banking tools to steal user account balances. The attack chain requires no advanced technical knowledge, server infrastructure, or monetary investment. The findings reveal a critical security gap in the emerging MCP ecosystem: while individual servers may appear trustworthy, their combination creates unexpected cross-server attack surfaces. Unlike traditional cybersecurity threats that assume sophisticated adversaries, our research shows that the barrier to entry for MCP-based attacks is alarmingly low. A threat actor with undergraduate-level Python knowledge can craft convincing social engineering attacks that exploit the implicit trust relationships MCP establishes between AI agents and tool providers. This work contributes to the nascent field of MCP security by demonstrating that current MCP implementations allow trivial cross-server attacks and proposing both immediate mitigations and protocol improvements to secure this emerging ecosystem.

摘要: 模型上下文协议（HCP）代表了人工智能工具集成的重大进步，实现了人工智能代理和外部服务之间的无缝通信。然而，这种连接性引入了新的攻击载体，而这些载体在很大程度上尚未被探索。本文展示了简单的威胁行为者，仅需要基本的编程技能和免费的网络工具，如何利用LCP的信任模型来泄露敏感的财务数据。我们提出了一种概念验证攻击，其中恶意天气HCP服务器伪装成良性功能，发现并利用合法的银行工具来窃取用户帐户余额。攻击链不需要先进的技术知识、服务器基础设施或金钱投资。研究结果揭示了新兴的LCP生态系统中的一个关键安全漏洞：虽然单个服务器可能看起来值得信赖，但它们的组合会造成意想不到的跨服务器攻击表面。与假设复杂对手的传统网络安全威胁不同，我们的研究表明，基于MPP的攻击的进入门槛低得惊人。具有本科生Python知识的威胁参与者可以利用LCP在人工智能代理和工具提供商之间建立的隐性信任关系来策划令人信服的社会工程攻击。这项工作通过证明当前的LCP实施允许轻微的跨服务器攻击，并提出立即缓解措施和协议改进来保护这个新兴生态系统，从而为LCP安全的新兴领域做出了贡献。



## **47. Cyber-attack TTP analysis for EPES systems**

EPES系统的网络攻击TTP分析 cs.NI

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2302.09164v2) [paper-pdf](http://arxiv.org/pdf/2302.09164v2)

**Authors**: Alexios Lekidis

**Abstract**: The electrical grid consists of legacy systems that were built with no security in mind. As we move towards the Industry 4.0 area though, a high-degree of automation and connectivity provides: 1) fast and flexible configuration and updates as well as 2) easier maintenance and handling of mis-configurations and operational errors. Even though considerations are present about the security implications of the Industry 4.0 era in the electrical grid, electricity stakeholders deem their infrastructures as secure since they are isolated and allow no external connections. However, external connections are not the only security risk for electrical utilities. The Tactics, Techniques and Procedures (TTPs) that are employed by adversaries to perform cyber-attack towards the critical Electrical Power and Energy System (EPES) infrastructures are gradually becoming highly advanced and sophisticated. In this article, we elaborate on these techniques and demonstrate them in a Power Plant of a major utility company within the Greek area. The demonstrated TTPs allow exploiting and executing remote commands in smart meters as well as Programmable Logic Controllers (PLCs) that are responsible for the power generator operation.

摘要: 电网由遗留系统组成，这些系统在构建时没有考虑到安全性。然而，随着我们向工业4.0领域迈进，高度自动化和连接性提供了：1）快速灵活的配置和更新，以及2）更容易的维护和处理错误配置和操作错误。尽管人们考虑到工业4.0时代对电网的安全影响，但电力利益相关者认为他们的基础设施是安全的，因为它们是孤立的，不允许外部连接。然而，外部连接并不是电力设施的唯一安全风险。对手用来对关键电力和能源系统（EPES）基础设施进行网络攻击的战术、技术和程序（TTP）正在逐渐变得高度先进和复杂。在本文中，我们详细介绍了这些技术，并在希腊地区一家主要公用事业公司的发电厂进行了演示。演示的TTP允许在智能电表以及负责发电机操作的可编程逻辑控制器（PLC）中利用和执行远程命令。



## **48. FedBAP: Backdoor Defense via Benign Adversarial Perturbation in Federated Learning**

FedBAP：通过联邦学习中的良性对抗扰动进行后门防御 cs.CR

Accepted to ACM Multimedia 2025

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.21177v1) [paper-pdf](http://arxiv.org/pdf/2507.21177v1)

**Authors**: Xinhai Yan, Libing Wu, Zhuangzhuang Zhang, Bingyi Liu, Lijuan Huo, Jing Wang

**Abstract**: Federated Learning (FL) enables collaborative model training while preserving data privacy, but it is highly vulnerable to backdoor attacks. Most existing defense methods in FL have limited effectiveness due to their neglect of the model's over-reliance on backdoor triggers, particularly as the proportion of malicious clients increases. In this paper, we propose FedBAP, a novel defense framework for mitigating backdoor attacks in FL by reducing the model's reliance on backdoor triggers. Specifically, first, we propose a perturbed trigger generation mechanism that creates perturbation triggers precisely matching backdoor triggers in location and size, ensuring strong influence on model outputs. Second, we utilize these perturbation triggers to generate benign adversarial perturbations that disrupt the model's dependence on backdoor triggers while forcing it to learn more robust decision boundaries. Finally, we design an adaptive scaling mechanism to dynamically adjust perturbation intensity, effectively balancing defense strength and model performance. The experimental results demonstrate that FedBAP reduces the attack success rates by 0.22%-5.34%, 0.48%-6.34%, and 97.22%-97.6% under three types of backdoor attacks, respectively. In particular, FedBAP demonstrates outstanding performance against novel backdoor attacks.

摘要: 联合学习（FL）支持协作模型训练，同时保护数据隐私，但它极易受到后门攻击。FL中大多数现有的防御方法的有效性有限，因为它们忽视了模型对后门触发器的过度依赖，特别是随着恶意客户端比例的增加。在本文中，我们提出了FedBAP，这是一种新型防御框架，用于通过减少模型对后门触发器的依赖来减轻FL中的后门攻击。具体来说，首先，我们提出了一种扰动触发器生成机制，该机制创建在位置和大小上与后门触发器精确匹配的扰动触发器，确保对模型输出产生强大影响。其次，我们利用这些扰动触发器来生成良性的对抗性扰动，从而破坏模型对后门触发器的依赖，同时迫使其学习更稳健的决策边界。最后，我们设计了自适应缩放机制来动态调整扰动强度，有效平衡防御强度和模型性能。实验结果表明，FedBAP在三种后门攻击下，攻击成功率分别降低了0.22%-5.34%、0.48%-6.34%和97.22%-97.6%。特别是，FedBAP在对抗新型后门攻击方面表现出色。



## **49. Enhancing IoT Intrusion Detection Systems through Adversarial Training**

通过对抗培训增强物联网入侵检测系统 cs.ET

6 pages

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.19739v1) [paper-pdf](http://arxiv.org/pdf/2507.19739v1)

**Authors**: Karma Gurung, Ashutosh Ghimire, Fathi Amsaad

**Abstract**: The augmentation of Internet of Things (IoT) devices transformed both automation and connectivity but revealed major security vulnerabilities in networks. We address these challenges by designing a robust intrusion detection system (IDS) to detect complex attacks by learning patterns from the NF-ToN-IoT v2 dataset. Intrusion detection has a realistic testbed through the dataset's rich and high-dimensional features. We combine distributed preprocessing to manage the dataset size with Fast Gradient Sign Method (FGSM) adversarial attacks to mimic actual attack scenarios and XGBoost model adversarial training for improved system robustness. Our system achieves 95.3% accuracy on clean data and 94.5% accuracy on adversarial data to show its effectiveness against complex threats. Adversarial training demonstrates its potential to strengthen IDS against evolving cyber threats and sets the foundation for future studies. Real-time IoT environments represent a future deployment opportunity for these systems, while extensions to detect emerging threats and zero-day vulnerabilities would enhance their utility.

摘要: 物联网（IOT）设备的增强改变了自动化和连接性，但也暴露了网络中的重大安全漏洞。我们通过设计一个强大的入侵检测系统（IDS）来解决这些挑战，通过从NF-ToN-IOT v2数据集学习模式来检测复杂的攻击。入侵检测通过数据集丰富且多维的特征拥有一个现实的测试平台。我们将管理数据集大小的分布式预处理与模拟实际攻击场景的快速梯度符号法（FGSM）对抗性攻击和XGBOP模型对抗性训练相结合，以提高系统稳健性。我们的系统在干净数据上实现了95.3%的准确性，在对抗性数据上实现了94.5%的准确性，以显示其应对复杂威胁的有效性。对抗性培训展示了其加强IDS抵御不断变化的网络威胁的潜力，并为未来的研究奠定了基础。实时物联网环境代表了这些系统的未来部署机会，而检测新兴威胁和零日漏洞的扩展将增强其实用性。



## **50. BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation**

BadVideo：针对文本转视频生成的秘密后门攻击 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2504.16907v2) [paper-pdf](http://arxiv.org/pdf/2504.16907v2)

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.

摘要: 文本转视频（T2 V）生成模型迅速发展，并在娱乐、教育和营销等领域获得了广泛应用。然而，这些模型的对抗漏洞仍然很少被探讨。我们观察到，在T2 V生成任务中，生成的视频通常包含大量文本提示中未明确指定的冗余信息，例如环境元素，次要对象和其他细节，为恶意攻击者嵌入隐藏的有害内容提供了机会。利用这种固有的冗余，我们引入了BadVideo，这是第一个为T2 V一代量身定制的后门攻击框架。我们的攻击重点是通过两个关键策略设计目标对抗输出：（1）时空合成，结合不同的时空特征来编码恶意信息;（2）动态元素转换，引入冗余元素随着时间的推移的转换以传达恶意信息。基于这些策略，攻击者的恶意目标与用户的文本指令无缝集成，提供高度的隐蔽性。此外，通过利用视频的时间维度，我们的攻击成功地规避了主要分析单个帧内空间信息的传统内容审核系统。大量实验表明，BadVideo在保留原始语义并在干净输入上保持出色的性能的同时实现了很高的攻击成功率。总体而言，我们的工作揭示了T2 V模型的对抗脆弱性，并引起人们对潜在风险和滥用的关注。我们的项目页面位于https://wrt2000.github.io/BadVideo2025/。



