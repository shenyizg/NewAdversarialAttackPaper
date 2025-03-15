# Latest Adversarial Attack Papers
**update at 2025-03-15 11:03:11**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.

摘要: 尽管在开源的大型视觉语言模型(LVLM)上性能很好，但基于传输的定向攻击对黑盒商业LVLM往往失败。分析失败的对抗性扰动发现，学习的扰动通常源于均匀分布，缺乏明确的语义细节，导致意外响应。这种严重的语义信息缺失导致商业LVLM要么完全忽略该扰动，要么曲解其嵌入的语义，从而导致攻击失败。为了克服这些问题，我们注意到，识别核心语义对象是用各种数据集和方法训练的模型的关键目标。这种洞察力激发了我们的方法，通过在局部区域内编码显式的语义细节来细化语义清晰度，从而确保互操作性和捕获更细粒度的特征，并通过将修改集中在语义丰富的区域而不是统一地应用它们。为了实现这一点，我们提出了一种简单而高效的解决方案：在每个优化步骤中，通过控制纵横比和比例来随机裁剪敌对图像，调整大小，然后在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们的对抗性例子使用聚焦于关键区域的局部聚集扰动来制作，表现出出奇地好的可转换性，可以用于商业LVLM，包括GPT-4.5、GPT-40、Gemini-2.0-Flash、Claude-3.5-十四行诗、Claude-3.7-十四行诗，甚至像o1、Claude-3.7-Think和Gemini-2.0-Flash-Think这样的推理模型。我们的方法在GPT-4.5、40o和o1上的成功率超过90%，大大超过了之前所有最先进的攻击方法。我们在不同配置和训练代码下的优化对抗性示例可在https://github.com/VILA-Lab/M-Attack.获得



## **2. Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

组织病理学中鲁棒视觉模型的分层自监督对抗训练 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10629v1) [paper-pdf](http://arxiv.org/pdf/2503.10629v1)

**Authors**: Hashmat Shadab Malik, Shahina Kunhimon, Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Adversarial attacks pose significant challenges for vision models in critical fields like healthcare, where reliability is essential. Although adversarial training has been well studied in natural images, its application to biomedical and microscopy data remains limited. Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images, where patient-slide-patch relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised Adversarial Training (HSAT), which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it into adversarial training for enhanced robustness. We evaluate HSAT on multiclass histopathology dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a new benchmark for adversarial training in this domain, paving the way for more robust models. Our Code for training and evaluation is available at https://github.com/HashmatShadab/HSAT.

摘要: 对抗性攻击对医疗保健等关键领域的视觉模型构成了重大挑战，在这些领域，可靠性至关重要。虽然对抗性训练已经在自然图像中得到了很好的研究，但它在生物医学和显微数据中的应用仍然有限。现有的自我监督的对抗性训练方法忽略了组织病理学图像的层次结构，其中患者-切片-补丁关系提供了有价值的区分信号。为了解决这一问题，我们提出了分层自监督对抗训练(HSAT)，它利用这些性质利用多级对比学习来构造对抗实例，并将其整合到对抗训练中以增强稳健性。我们在多类组织病理学数据集OpenSRH上对HSAT进行了评估，结果表明HSAT的性能优于生物医学和自然图像领域的现有方法。HSAT增强了健壮性，在白盒设置中实现了54.31%的平均增益，并将黑盒设置中的性能下降到了3-4%，而基准设置为25-30%。这些结果为这一领域的对抗性训练设定了新的基准，为更稳健的模型铺平了道路。我们的培训和评估准则可在https://github.com/HashmatShadab/HSAT.上获得



## **3. Towards Class-wise Robustness Analysis**

走向班级稳健性分析 cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2411.19853v2) [paper-pdf](http://arxiv.org/pdf/2411.19853v2)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.

摘要: 虽然深度神经网络在解决许多下游任务方面非常成功，但由于其对域转移的敏感性，如常见的腐败和敌对攻击，其在现实生活场景中的应用受到限制。对抗性例子和数据破坏的存在大大降低了深度分类模型的性能。研究人员在开发稳健的神经体系结构以支持深度分类器的决策方面取得了很大进展。然而，这些工作大多依赖于有效的对抗性训练方法，并且主要关注整体模型的稳健性，而忽略了类之间的稳健性差异，这是至关重要的。利用健壮性较弱的类是攻击者愚弄图像识别模型的潜在途径。因此，本研究通过研究反向训练的稳健分类模型的类对类偏差，以了解它们的潜在空间结构，并分析它们的强弱类性质。我们进一步评估了类对常见的腐败和敌意攻击的健壮性，认识到类的脆弱性超出了特定类的正确分类的数量。我们发现，作为特定目标类的类的误报数量显著影响其易受攻击的程度。通过我们对班级假阳性分数的分析，我们评估了每个班级对错误分类的易感性的公平评估。



## **4. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

TH-Bench：通过机器生成文本检测器上人性化人工智能文本来评估逃避攻击 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.

摘要: 随着大型语言模型(LLM)的发展，机器生成文本(MGTS)变得越来越流畅、高质量和信息丰富。现有的大范围MGT探测器旨在识别MGT，以防止抄袭和错误信息的传播。然而，攻击者试图使MGTS人性化以躲避检测(称为躲避攻击)，这只需要进行少量修改即可绕过MGT检测器。遗憾的是，现有的攻击通常缺乏统一和全面的评估框架，因为它们是使用不同的实验设置、模型架构和数据集进行评估的。为了填补这一空白，我们引入了文本人性化基准(TH-BENCH)，这是第一个评估针对MGT检测器的躲避攻击的综合基准。TH-BENCH从三个关键维度对攻击进行评估：规避效率、文本质量和计算开销。我们的广泛实验评估了6种针对6个数据集的13个MGT检测器的最先进攻击，这些攻击跨越19个域，由11个广泛使用的LLM生成。我们的发现表明，没有一次逃避攻击在所有三个维度上都表现出色。通过深入的分析，我们突出了不同攻击的优势和局限性。更重要的是，我们确定了三个维度之间的权衡，并提出了两个优化见解。通过初步实验，验证了它们的正确性和有效性，为以后的研究提供了潜在的方向。



## **5. I Can Tell Your Secrets: Inferring Privacy Attributes from Mini-app Interaction History in Super-apps**

我可以告诉你的秘密：从超级应用中的迷你应用交互历史记录推断隐私属性 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10239v1) [paper-pdf](http://arxiv.org/pdf/2503.10239v1)

**Authors**: Yifeng Cai, Ziqi Zhang, Mengyu Yao, Junlin Liu, Xiaoke Zhao, Xinyi Fu, Ruoyu Li, Zhe Li, Xiangqun Chen, Yao Guo, Ding Li

**Abstract**: Super-apps have emerged as comprehensive platforms integrating various mini-apps to provide diverse services. While super-apps offer convenience and enriched functionality, they can introduce new privacy risks. This paper reveals a new privacy leakage source in super-apps: mini-app interaction history, including mini-app usage history (Mini-H) and operation history (Op-H). Mini-H refers to the history of mini-apps accessed by users, such as their frequency and categories. Op-H captures user interactions within mini-apps, including button clicks, bar drags, and image views. Super-apps can naturally collect these data without instrumentation due to the web-based feature of mini-apps. We identify these data types as novel and unexplored privacy risks through a literature review of 30 papers and an empirical analysis of 31 super-apps. We design a mini-app interaction history-oriented inference attack (THEFT), to exploit this new vulnerability. Using THEFT, the insider threats within the low-privilege business department of the super-app vendor acting as the adversary can achieve more than 95.5% accuracy in inferring privacy attributes of over 16.1% of users. THEFT only requires a small training dataset of 200 users from public breached databases on the Internet. We also engage with super-app vendors and a standards association to increase industry awareness and commitment to protect this data. Our contributions are significant in identifying overlooked privacy risks, demonstrating the effectiveness of a new attack, and influencing industry practices toward better privacy protection in the super-app ecosystem.

摘要: 超级应用已经成为整合各种小应用以提供多样化服务的综合平台。虽然超级应用程序提供了便利和丰富的功能，但它们也可能带来新的隐私风险。揭示了超级应用中一个新的隐私泄漏源：小应用交互历史，包括小应用使用历史(Mini-H)和操作历史(Op-H)。Mini-H指的是用户访问小应用的历史，比如它们的频率和类别。OP-H捕获迷你应用程序中的用户交互，包括按钮点击、栏拖动和图像查看。由于小应用程序的基于网络的功能，超级应用程序可以自然地收集这些数据，而不需要工具。我们通过对30篇论文的文献回顾和对31个超级应用程序的实证分析，将这些数据类型确定为新的和未探索的隐私风险。我们设计了一个面向交互历史的迷你APP推理攻击(盗窃)，来利用这个新的漏洞。使用盗窃，作为对手的超级应用供应商低权限业务部门的内部威胁可以达到95.5%以上的准确率，推断出16.1%以上的用户的隐私属性。盗窃只需要一个由200名用户组成的小型培训数据集，这些用户来自互联网上被入侵的公共数据库。我们还与超级应用程序供应商和标准协会接触，以提高行业意识和承诺保护这些数据。我们的贡献在识别被忽视的隐私风险、展示新攻击的有效性以及影响行业做法以在超级应用生态系统中更好地保护隐私方面具有重要意义。



## **6. Robustness Tokens: Towards Adversarial Robustness of Transformers**

稳健代币：走向变形金刚的对抗稳健 cs.LG

This paper has been accepted for publication at the European  Conference on Computer Vision (ECCV), 2024

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10191v1) [paper-pdf](http://arxiv.org/pdf/2503.10191v1)

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.

摘要: 最近，大型预训练基础模型已被机器学习从业者广泛采用，用于多种任务。鉴于此类模型是公开的，依赖于它们作为下游任务的主干模型可能会导致对使用相同公共模型设计的对抗攻击的高度脆弱性。在这项工作中，我们提出了鲁棒性令牌，这是一种特定于Transformer架构的新颖方法，可以以低计算要求微调一些额外的私有令牌，而不是像传统对抗训练中那样调整模型参数。我们表明，鲁棒性令牌使Vision Transformer模型对白盒对抗攻击的鲁棒性明显更强，同时还保留了原始的下游性能。



## **7. Can't Slow me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices**

无法让我慢下来：学习强大且硬件自适应的对象检测器来对抗边缘设备的延迟攻击 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.02171v2) [paper-pdf](http://arxiv.org/pdf/2412.02171v2)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.

摘要: 目标检测是自动驾驶、增强现实和供应链管理等许多实时下游应用的基本使能。然而，神经网络的算法主干对系统输入中的不可察觉的扰动是脆弱的，这种扰动通常被称为误分类攻击。以实时处理能力为目标，最近报道了一类新的延迟攻击。它们通过在后处理模块中创建计算瓶颈来利用对象检测器中的新攻击面，从而导致级联故障并使实时下游任务处于危险之中。在这项工作中，我们初步尝试通过背景专注的对手训练来防御这种攻击，该训练也认识到潜在的硬件能力。我们首先在系统级将延迟攻击与跨不同类型的GPU设备的硬件容量联系起来。基于特定的对抗性行为，我们利用客观性损失作为代理，在对抗性训练流水线中加入背景注意，在干净和健壮的准确率之间取得合理的平衡。广泛的实验证明了在Jetson Orin NX上将实时处理能力从13美元FPS恢复到43美元FPS的防御效果，并在干净和健壮的准确性之间进行了更好的权衡。



## **8. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

可转移对抗攻击的预算驱动对比学习 cs.CV

Accepted to ECCV 2024 (Oral), Project Page:  https://PDCL-Attack.github.io

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2407.20657v2) [paper-pdf](http://arxiv.org/pdf/2407.20657v2)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.

摘要: 最近的视觉语言基础模型，如CLIP，在学习表征方面表现出了优越的能力，这些表征可以跨不同范围的下游任务和领域转移。随着如此强大的模型的出现，有效地利用它们的能力来处理具有挑战性的愿景任务变得至关重要。另一方面，只有少数工作专注于设计能够很好地移植到未知领域和模型体系结构的对抗性例子。本文提出了一种新的转移攻击方法PDCL-Attack，该方法利用CLIP模型来增强基于产生式模型的攻击框架产生的敌意扰动的可转移性。具体地说，我们通过利用文本的语义表征能力，特别是从输入图像的基本事实类标签来制定有效的提示驱动的特征指导。据我们所知，我们是第一个引入快速学习来增强可转移的生成性攻击的。在各种跨域和跨模型环境下进行的广泛实验验证了我们的方法，证明了它比最先进的方法更优越。



## **9. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

对抗非中心扰动：用$l_1$-模估计精确恢复线性系统 math.OC

8 pages, 2 figures

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2410.03218v5) [paper-pdf](http://arxiv.org/pdf/2410.03218v5)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.

摘要: 本文研究一般情况下的线性系统辨识问题，其中扰动是亚高斯的，相关的，可能是对抗性的。首先，我们考虑了具有非中心(非零均值)扰动的情况，对于这种情况，普通的最小二乘(OLS)方法不能正确地辨识系统。我们证明了在每个扰动具有相等的正负概率的条件下，$L_1$-范数估计量能够准确地辨识系统。这一条件限制了每个扰动的符号，但允许其大小任意。其次，在攻击次数偶然发生但攻击值的分布是任意的情况下，我们考虑了每次扰动都是对抗性的情况。证明了当给定时刻发生攻击的概率小于0.5时，当每次攻击跨越期望的整个空间时，$L_1$-范数估计对任何对抗性非中心扰动都是有效的，并且在有限时间内实现了精确的恢复。这些结果为在安全关键系统中有效防御任意规模的非中心攻击铺平了道路。



## **10. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

AnywhereDoor：对象检测的多目标后门攻击 cs.CR

This work was intended as a replacement of arXiv:2411.14243 and any  subsequent updates will appear there

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.06529v2) [paper-pdf](http://arxiv.org/pdf/2503.06529v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.

摘要: 随着对象检测成为许多安全关键型应用程序不可或缺的一部分，了解其漏洞至关重要。尤其是后门攻击，通过在受害者模型中植入隐藏的触发器，构成了严重的威胁，攻击者稍后可以利用这些触发器在推理过程中诱导恶意行为。然而，目前的理解仅限于单目标攻击，即攻击者必须在训练前定义一个固定的恶意行为(目标)，这使得推理时间适应性变得不可能。考虑到目标检测(包括目标存在预测、包围盒估计和分类)的大输出空间，灵活的推理时间模型控制的可行性仍未被探索。本文介绍了Anywhere Door，一种用于目标检测的多目标后门攻击。一旦被植入，Anywhere Door允许对手在所有对象类或特定对象类中让对象消失、捏造新对象或错误标记对象，提供前所未有的控制程度。这种灵活性是由三项关键创新实现的：(I)客观解缠，以扩大受支持目标的数量；(Ii)触发马赛克，以确保即使针对基于区域的探测器也具有稳健性；以及(Iii)战略批处理，以解决阻碍操纵的对象级数据失衡。广泛的实验表明，Anywhere Door给予攻击者高度的控制，与采用现有方法进行这种灵活的控制相比，攻击成功率提高了26%。



## **11. The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions**

LLM生成的合成数据用于在线政治讨论中立场检测的力量 cs.CL

ICLR 2025 Spotlight

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2406.12480v2) [paper-pdf](http://arxiv.org/pdf/2406.12480v2)

**Authors**: Stefan Sylvius Wagner, Maike Behrendt, Marc Ziegele, Stefan Harmeling

**Abstract**: Stance detection holds great potential to improve online political discussions through its deployment in discussion platforms for purposes such as content moderation, topic summarization or to facilitate more balanced discussions. Typically, transformer-based models are employed directly for stance detection, requiring vast amounts of data. However, the wide variety of debate topics in online political discussions makes data collection particularly challenging. LLMs have revived stance detection, but their online deployment in online political discussions faces challenges like inconsistent outputs, biases, and vulnerability to adversarial attacks. We show how LLM-generated synthetic data can improve stance detection for online political discussions by using reliable traditional stance detection models for online deployment, while leveraging the text generation capabilities of LLMs for synthetic data generation in a secure offline environment. To achieve this, (i) we generate synthetic data for specific debate questions by prompting a Mistral-7B model and show that fine-tuning with the generated synthetic data can substantially improve the performance of stance detection, while remaining interpretable and aligned with real world data. (ii) Using the synthetic data as a reference, we can improve performance even further by identifying the most informative samples in an unlabelled dataset, i.e., those samples which the stance detection model is most uncertain about and can benefit from the most. By fine-tuning with both synthetic data and the most informative samples, we surpass the performance of the baseline model that is fine-tuned on all true labels, while labelling considerably less data.

摘要: 立场检测具有巨大的潜力，可以通过在讨论平台上部署立场检测来改进在线政治讨论，目的是为了内容审查、话题总结或促进更平衡的讨论。通常，基于变压器的模型直接用于姿态检测，需要大量数据。然而，在线政治讨论中的辩论话题种类繁多，这使得数据收集尤其具有挑战性。LLM恢复了立场检测，但它们在在线政治讨论中的在线部署面临着输出不一致、偏见和易受对手攻击等挑战。我们展示了LLM生成的合成数据如何通过使用可靠的传统立场检测模型进行在线部署来改进在线政治讨论的立场检测，同时利用LLMS的文本生成能力在安全的离线环境中生成合成数据。为了实现这一点，(I)我们通过提示Mistral-7B模型为特定辩论问题生成合成数据，并表明使用生成的合成数据进行微调可以显著提高姿势检测的性能，同时保持可解释并与真实世界的数据保持一致。(Ii)使用合成数据作为参考，我们可以通过在未标记的数据集中识别信息量最大的样本来进一步提高性能，即姿态检测模型最不确定且最能受益的样本。通过对合成数据和信息量最大的样本进行微调，我们超过了在所有真实标签上进行微调的基线模型的性能，而标记的数据要少得多。



## **12. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.

摘要: 大型语言模型(LLM)最近在时间序列预测方面显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验包括LLMTime与GPT-3.5、GPT-4、Llama和Mistral、TimeGPT和TimeLLM的模型，实验表明，对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。代码存储库可在https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.上找到



## **13. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2409.11295v5) [paper-pdf](http://arxiv.org/pdf/2409.11295v5)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.

摘要: 多面手网络代理在自主完成真实网站上的各种任务方面表现出了非凡的潜力，显著提高了人类的生产力。然而，预订机票等网络任务通常涉及用户的PII，如果网络代理意外地与受影响的网站交互，可能会面临潜在的隐私风险，这种情况在文献中基本上仍未探讨。在这项工作中，我们通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们给出了一个现实的网站攻击威胁模型，其中我们考虑了两个敌对目标：窃取用户的特定PII或整个用户请求。然后，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。EIA注入恶意内容，旨在很好地适应代理运行的环境，我们的工作特别针对Web环境中的隐私场景实例化了EIA。我们从Mind2Web收集了177个动作步骤，涉及现实网站上不同的PII类别，并使用迄今最有能力的通才Web代理框架之一进行了实验。结果表明，EIA在窃取特定PII请求时获得了高达70%的ASR，对于完整的用户请求达到了16%的ASR。此外，通过访问隐蔽性和试验防御系统提示，我们表明EIA很难检测和缓解。值得注意的是，没有很好地适应网页的攻击可以通过人工检查来检测，这导致了我们关于安全性和自主性之间的权衡的讨论。然而，额外的攻击者的努力可能会使EIA无缝适应，使这种监督无效。因此，我们进一步讨论了网站部署前和部署后阶段的防御，而不依赖于人的监督，并呼吁更先进的防御策略。



## **14. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

论科尔莫戈洛夫-阿诺德网络的鲁棒性：对抗的视角 cs.CV

Accepted at TMLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2408.13809v3) [paper-pdf](http://arxiv.org/pdf/2408.13809v3)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.

摘要: Kolmogorov-Arnold网络(KANS)是最近出现的一种新的函数逼近方法，在各个领域显示出巨大的潜力。尽管它们在理论上有希望，但KANS在对抗条件下的健壮性尚未得到彻底的检验。在这篇文章中，我们探索了KANS的对抗稳健性，特别关注图像分类任务。我们评估了人工神经网络对标准白盒和黑盒对抗攻击的性能，比较了它们与已建立的神经网络结构的弹性。我们的实验评估涵盖了各种标准图像分类基准数据集，并研究了三种规模的完全连接和卷积神经网络结构：小型、中型和大型。我们的结论是，小型和中型KAN(无论是完全连接的还是卷积的)并不总是比它们的标准对应产品更健壮，但总的来说，大型KAN更健壮。这项对对抗性情景下的KANS的全面评估提供了第一次对KAN安全性的深入分析，为这一新兴领域的未来研究奠定了基础。



## **15. ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection**

ICMarks：用于集成电路物理设计IP保护的稳健水印框架 cs.CR

accept to TCAD (IEEE Transactions on Computer-Aided Design of  Integrated Circuits and Systems)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2404.18407v2) [paper-pdf](http://arxiv.org/pdf/2404.18407v2)

**Authors**: Ruisi Zhang, Rachel Selina Rajarathnam, David Z. Pan, Farinaz Koushanfar

**Abstract**: Physical design watermarking on contemporary integrated circuit (IC) layout encodes signatures without considering the dense connections and design constraints, which could lead to performance degradation on the watermarked products. This paper presents ICMarks, a quality-preserving and robust watermarking framework for modern IC physical design. ICMarks embeds unique watermark signatures during the physical design's placement stage, thereby authenticating the IC layout ownership. ICMarks's novelty lies in (i) strategically identifying a region of cells to watermark with minimal impact on the layout performance and (ii) a two-level watermarking framework for augmented robustness toward potential removal and forging attacks. Extensive evaluations on benchmarks of different design objectives and sizes validate that ICMarks incurs no wirelength and timing metrics degradation, while successfully proving ownership. Furthermore, we demonstrate ICMarks is robust against two major watermarking attack categories, namely, watermark removal and forging attacks; even if the adversaries have prior knowledge of the watermarking schemes, the signatures cannot be removed without significantly undermining the layout quality.

摘要: 现代集成电路(IC)版图上的物理设计水印在未考虑密集连接和设计约束的情况下对签名进行编码，这可能会导致水印产品的性能下降。本文提出了ICMarks，一种用于现代集成电路物理设计的质量保持和健壮的水印框架。ICMarks在物理设计的放置阶段嵌入唯一的水印签名，从而验证IC版图所有权。ICMarks的创新之处在于：(I)战略性地识别要添加水印的单元区域，而对布局性能的影响最小；(Ii)两级水印框架，可增强对潜在删除和伪造攻击的稳健性。对不同设计目标和规模的基准的广泛评估证实，ICMarks在成功证明所有权的同时，不会导致有线长度和计时指标降级。此外，我们证明了ICMarks对两种主要的水印攻击，即水印移除和伪造攻击具有很强的鲁棒性；即使攻击者事先知道水印方案，也无法在不显著损害布局质量的情况下移除签名。



## **16. Enhancing Adversarial Example Detection Through Model Explanation**

通过模型解释增强对抗示例检测 cs.CR

5 pages, 1 figure

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09735v1) [paper-pdf](http://arxiv.org/pdf/2503.09735v1)

**Authors**: Qian Ma, Ziping Ye

**Abstract**: Adversarial examples are a major problem for machine learning models, leading to a continuous search for effective defenses. One promising direction is to leverage model explanations to better understand and defend against these attacks. We looked at AmI, a method proposed by a NeurIPS 2018 spotlight paper that uses model explanations to detect adversarial examples. Our study shows that while AmI is a promising idea, its performance is too dependent on specific settings (e.g., hyperparameter) and external factors such as the operating system and the deep learning framework used, and such drawbacks limit AmI's practical usage. Our findings highlight the need for more robust defense mechanisms that are effective under various conditions. In addition, we advocate for a comprehensive evaluation framework for defense techniques.

摘要: 对抗性示例是机器学习模型的一个主要问题，导致人们不断寻找有效的防御。一个有希望的方向是利用模型解释来更好地理解和防御这些攻击。我们研究了AmI，这是一种由NeurIPS 2018年焦点论文提出的方法，它使用模型解释来检测对抗性示例。我们的研究表明，虽然AmI是一个有前途的想法，但其性能过于依赖于特定设置（例如，超参数）以及所使用的操作系统和深度学习框架等外部因素，这些缺陷限制了AmI的实际使用。我们的研究结果强调了需要更强大的、在各种条件下有效的防御机制。此外，我们倡导对防御技术建立全面的评估框架。



## **17. All Your Knowledge Belongs to Us: Stealing Knowledge Graphs via Reasoning APIs**

您所有的知识都属于我们：通过推理API窃取知识图 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09727v1) [paper-pdf](http://arxiv.org/pdf/2503.09727v1)

**Authors**: Zhaohan Xi

**Abstract**: Knowledge graph reasoning (KGR), which answers complex, logical queries over large knowledge graphs (KGs), represents an important artificial intelligence task with a range of applications. Many KGs require extensive domain expertise and engineering effort to build and are hence considered proprietary within organizations and enterprises. Yet, spurred by their commercial and research potential, there is a growing trend to make KGR systems, (partially) built upon private KGs, publicly available through reasoning APIs.   The inherent tension between maintaining the confidentiality of KGs while ensuring the accessibility to KGR systems motivates our study of KG extraction attacks: the adversary aims to "steal" the private segments of the backend KG, leveraging solely black-box access to the KGR API. Specifically, we present KGX, an attack that extracts confidential sub-KGs with high fidelity under limited query budgets. At a high level, KGX progressively and adaptively queries the KGR API and integrates the query responses to reconstruct the private sub-KG. This extraction remains viable even if any query responses related to the private sub-KG are filtered. We validate the efficacy of KGX against both experimental and real-world KGR APIs. Interestingly, we find that typical countermeasures (e.g., injecting noise into query responses) are often ineffective against KGX. Our findings suggest the need for a more principled approach to developing and deploying KGR systems, as well as devising new defenses against KG extraction attacks.

摘要: 知识图推理(KGR)是人工智能的一项重要任务，具有广泛的应用前景，它能够回答大型知识图上复杂的逻辑查询问题。许多KG需要广泛的领域专业知识和工程工作来构建，因此在组织和企业中被认为是专有的。然而，在它们的商业和研究潜力的刺激下，有一种日益增长的趋势，使KGR系统(部分)建立在私有KG之上，通过推理API公开可用。维护KG的机密性和确保KGR系统的可访问性之间的内在矛盾促使我们研究KG提取攻击：对手的目标是通过仅利用对KGR API的黑盒访问来“窃取”后端KG的私有部分。具体地说，我们提出了KGX，这是一种在有限的查询预算下高保真地提取保密子KG的攻击。在较高级别上，KGX渐进地和自适应地查询KGR API，并集成查询响应以重建私有子KG。即使过滤了与私有子KG相关的任何查询响应，该提取仍然是可行的。我们在实验和现实世界的KGR API上验证了KGX的有效性。有趣的是，我们发现典型的对策(例如，向查询响应中注入噪声)通常对KGX无效。我们的发现表明，需要一种更有原则的方法来开发和部署KGR系统，以及设计新的防御KG提取攻击的方法。



## **18. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

针对辅助规模超分辨率的规模不变对抗攻击 cs.CV

17 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.04385v2) [paper-pdf](http://arxiv.org/pdf/2503.04385v2)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.

摘要: 局部连续图像函数(LIIF)的出现引起了任意尺度超分辨率(SR)技术的极大关注。然而，尽管固定尺度SR的脆弱性已经被评估，但基于连续表示的任意尺度SR对敌意攻击的稳健性仍然是一个值得进一步研究的领域。针对固定规模SR精心设计的敌意攻击是规模相关的，当应用于任意规模SR时，会导致耗时和内存消耗的问题。为了解决这一问题，我们提出了一种简单而有效的具有良好可转移性的“尺度不变”SR对抗攻击方法，称为SIAGT。具体地说，我们提出了利用连续表示的有限离散点来构造节省资源的攻击。此外，为了增强攻击的跨模型可转移性，我们还制定了一个坐标相关损失。该攻击可以显著恶化SR图像，同时给目标低分辨率(LR)图像带来不可察觉的失真。在三种流行的基于LIIF的SR方法和四个经典SR数据集上的实验表明，SIAGT具有良好的攻击性能和可转移性。



## **19. RESTRAIN: Reinforcement Learning-Based Secure Framework for Trigger-Action IoT Environment**

RESTRAIN：用于触发动作物联网环境的基于强化学习的安全框架 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09513v1) [paper-pdf](http://arxiv.org/pdf/2503.09513v1)

**Authors**: Md Morshed Alam, Lokesh Chandra Das, Sandip Roy, Sachin Shetty, Weichao Wang

**Abstract**: Internet of Things (IoT) platforms with trigger-action capability allow event conditions to trigger actions in IoT devices autonomously by creating a chain of interactions. Adversaries exploit this chain of interactions to maliciously inject fake event conditions into IoT hubs, triggering unauthorized actions on target IoT devices to implement remote injection attacks. Existing defense mechanisms focus mainly on the verification of event transactions using physical event fingerprints to enforce the security policies to block unsafe event transactions. These approaches are designed to provide offline defense against injection attacks. The state-of-the-art online defense mechanisms offer real-time defense, but extensive reliability on the inference of attack impacts on the IoT network limits the generalization capability of these approaches. In this paper, we propose a platform-independent multi-agent online defense system, namely RESTRAIN, to counter remote injection attacks at runtime. RESTRAIN allows the defense agent to profile attack actions at runtime and leverages reinforcement learning to optimize a defense policy that complies with the security requirements of the IoT network. The experimental results show that the defense agent effectively takes real-time defense actions against complex and dynamic remote injection attacks and maximizes the security gain with minimal computational overhead.

摘要: 具有触发动作功能的物联网(IoT)平台允许事件条件通过创建一系列交互来自主地触发物联网设备中的动作。攻击者利用这一链条相互作用，向物联网集线器恶意注入虚假事件条件，触发对目标物联网设备的未经授权的操作，以实施远程注入攻击。现有的防御机制主要集中在使用物理事件指纹来验证事件事务，以执行安全策略来阻止不安全的事件事务。这些方法旨在提供针对注入攻击的离线防御。最先进的在线防御机制提供了实时防御，但在推断攻击对物联网网络的影响方面存在广泛的可靠性，限制了这些方法的推广能力。本文提出了一种与平台无关的多智能体在线防御系统，即Restraint，用于在运行时对抗远程注入攻击。Restraint允许防御代理在运行时分析攻击操作，并利用强化学习来优化符合物联网网络安全要求的防御策略。实验结果表明，该防御代理能够有效地对复杂、动态的远程注入攻击采取实时防御行动，以最小的计算开销最大化安全收益。



## **20. Independence Tests for Language Models**

语言模型的独立性测试 cs.LG

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2502.12292v2) [paper-pdf](http://arxiv.org/pdf/2502.12292v2)

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.

摘要: 我们考虑以下问题：给定两个模型的权重，我们能否测试它们是否独立训练--即从独立的随机初始化？我们考虑两种设置：受约束和不受约束。在约束环境下，我们对模型结构和训练进行了假设，并提出了一族统计检验，这些检验相对于模型是从独立的随机初始化训练而来的零假设产生精确的p值。无论任何一个模型的训练数据的组成如何，这些p值都是有效的；我们通过在我们的假设下模拟每个模型的可交换副本，并将原始两个模型之间的权重和激活的各种相似性度量与这些副本进行比较来计算它们。我们报告了21对公开重量模型(总共210对)的p值，并正确识别了所有非独立模型对。我们的测试仍然有效，即使一个模型针对多个令牌进行了微调。在不受约束的设置中，我们不对训练过程做出假设，可以改变模型架构，并允许对抗性逃避攻击，以前的测试不再起作用。相反，我们提出了一种新的测试，它匹配两个模型之间的隐藏激活，并且对对抗性转换和模型体系结构的变化具有健壮性。该测试还可以进行本地化测试：识别模型的特定非独立组件。尽管我们不再从中获得确切的p值，但从经验上讲，我们发现它的行为像一个人，并可靠地识别非独立模型。值得注意的是，我们可以使用测试来识别一个模型从另一个模型派生的特定部分(例如，如何修剪Llama 3.1-8B以初始化Llama 3.2-3B，或如何在Mistral-7B和StriedHyena-7B之间共享层)，甚至从头开始重新训练任一模型的各个层都是稳健的。



## **21. Revisiting Medical Image Retrieval via Knowledge Consolidation**

通过知识整合重新审视医学图像检索 cs.CV

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09370v1) [paper-pdf](http://arxiv.org/pdf/2503.09370v1)

**Authors**: Yang Nan, Huichi Zhou, Xiaodan Xing, Giorgos Papanastasiou, Lei Zhu, Zhifan Gao, Alejandro F Fangi, Guang Yang

**Abstract**: As artificial intelligence and digital medicine increasingly permeate healthcare systems, robust governance frameworks are essential to ensure ethical, secure, and effective implementation. In this context, medical image retrieval becomes a critical component of clinical data management, playing a vital role in decision-making and safeguarding patient information. Existing methods usually learn hash functions using bottleneck features, which fail to produce representative hash codes from blended embeddings. Although contrastive hashing has shown superior performance, current approaches often treat image retrieval as a classification task, using category labels to create positive/negative pairs. Moreover, many methods fail to address the out-of-distribution (OOD) issue when models encounter external OOD queries or adversarial attacks. In this work, we propose a novel method to consolidate knowledge of hierarchical features and optimisation functions. We formulate the knowledge consolidation by introducing Depth-aware Representation Fusion (DaRF) and Structure-aware Contrastive Hashing (SCH). DaRF adaptively integrates shallow and deep representations into blended features, and SCH incorporates image fingerprints to enhance the adaptability of positive/negative pairings. These blended features further facilitate OOD detection and content-based recommendation, contributing to a secure AI-driven healthcare environment. Moreover, we present a content-guided ranking to improve the robustness and reproducibility of retrieval results. Our comprehensive assessments demonstrate that the proposed method could effectively recognise OOD samples and significantly outperform existing approaches in medical image retrieval (p<0.05). In particular, our method achieves a 5.6-38.9% improvement in mean Average Precision on the anatomical radiology dataset.

摘要: 随着人工智能和数字医学日益渗透到医疗系统中，强大的治理框架对于确保道德、安全和有效的实施至关重要。在这种背景下，医学图像检索成为临床数据管理的重要组成部分，在决策和保护患者信息方面发挥着至关重要的作用。现有的方法通常使用瓶颈特征来学习哈希函数，而瓶颈特征无法从混合嵌入中产生具有代表性的哈希码。尽管对比散列法表现出了优越的性能，但目前的方法通常将图像检索视为一项分类任务，使用类别标签来创建正/负对。此外，当模型遇到外部OOD查询或对手攻击时，许多方法无法解决分布外(OOD)问题。在这项工作中，我们提出了一种新的方法来整合层次特征和优化函数的知识。我们通过引入深度感知表示融合(DARF)和结构感知对比散列(SCH)来描述知识整合。DARF自适应地将浅层和深层表示融合到混合特征中，SCH融合图像指纹以增强正/负配对的适应性。这些混合功能进一步促进了OOD检测和基于内容的推荐，为安全的人工智能驱动的医疗环境做出了贡献。此外，为了提高检索结果的稳健性和可重复性，我们提出了一种基于内容的排序方法。综合评价结果表明，该方法能够有效识别OOD样本，并显著优于现有的医学图像检索方法(p<0.05)。特别是，我们的方法在解剖放射学数据集上的平均精度提高了5.6-38.9%。



## **22. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.

摘要: 将大型语言模型(LLM)集成到网络安全应用程序中提供了重大机遇，如增强威胁分析和恶意软件检测，但也可能带来重大风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMInstruct，这是一个包含54,928个指令-响应对的数据集，涵盖了网络安全任务，如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这包括从多个来源获取数据，将其过滤和组织成指令-响应对，并使其与现实世界的情景相一致，以增强其适用性。选择了7个开源LLM来测试CyberLLMInstruct的有用性：Phi 3 Mini 3.8B、Mistral 7B、Qwen 2.5 7B、Llama 3 8B、Llama 3.1 8B、Gema 2 9B和Llama 2 70B。在我们的主要示例中，我们使用OWASP TOP 10框架严格评估微调模型的安全性，发现微调降低了所有测试的LLM和每个对手攻击的安全弹性(例如，针对即时注入的Llama 3.1 8B的安全分数从0.95下降到0.15)。在我们的第二个示例中，我们展示了这些相同的微调模型在CyberMetric基准上也可以达到高达92.50%的准确率。这些发现突出了性能和安全之间的权衡，显示了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍然提高不同数据集和域的性能。复制数据集所需的所有脚本，以及复制我们的结果的例子和相关资源，将在论文被接受时提供。



## **23. Detecting and Preventing Data Poisoning Attacks on AI Models**

检测和预防对人工智能模型的数据中毒攻击 cs.CR

9 pages, 8 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09302v1) [paper-pdf](http://arxiv.org/pdf/2503.09302v1)

**Authors**: Halima I. Kure, Pradipta Sarkar, Ahmed B. Ndanusa, Augustine O. Nwajana

**Abstract**: This paper investigates the critical issue of data poisoning attacks on AI models, a growing concern in the ever-evolving landscape of artificial intelligence and cybersecurity. As advanced technology systems become increasingly prevalent across various sectors, the need for robust defence mechanisms against adversarial attacks becomes paramount. The study aims to develop and evaluate novel techniques for detecting and preventing data poisoning attacks, focusing on both theoretical frameworks and practical applications. Through a comprehensive literature review, experimental validation using the CIFAR-10 and Insurance Claims datasets, and the development of innovative algorithms, this paper seeks to enhance the resilience of AI models against malicious data manipulation. The study explores various methods, including anomaly detection, robust optimization strategies, and ensemble learning, to identify and mitigate the effects of poisoned data during model training. Experimental results indicate that data poisoning significantly degrades model performance, reducing classification accuracy by up to 27% in image recognition tasks (CIFAR-10) and 22% in fraud detection models (Insurance Claims dataset). The proposed defence mechanisms, including statistical anomaly detection and adversarial training, successfully mitigated poisoning effects, improving model robustness and restoring accuracy levels by an average of 15-20%. The findings further demonstrate that ensemble learning techniques provide an additional layer of resilience, reducing false positives and false negatives caused by adversarial data injections.

摘要: 本文研究了针对人工智能模型的数据中毒攻击这一关键问题，在不断发展的人工智能和网络安全领域，这是一个日益令人担忧的问题。随着先进技术系统在各个部门变得越来越普遍，对对抗攻击的强大防御机制的需要变得至关重要。这项研究旨在开发和评估检测和预防数据中毒攻击的新技术，重点放在理论框架和实际应用上。通过全面的文献综述，使用CIFAR-10和保险索赔数据集进行的实验验证，以及创新算法的开发，本文试图增强AI模型对恶意数据操纵的弹性。该研究探索了各种方法，包括异常检测、稳健优化策略和集成学习，以识别和缓解模型训练过程中有毒数据的影响。实验结果表明，数据中毒显著降低了模型的性能，在图像识别任务(CIFAR-10)中，分类准确率降低了27%，在欺诈检测模型(保险索赔数据集)中，分类精度降低了22%。提出的防御机制，包括统计异常检测和对抗性训练，成功地缓解了中毒效应，提高了模型的稳健性，并将准确率平均恢复了15%-20%。研究结果进一步表明，集成学习技术提供了一层额外的弹性，减少了由对抗性数据注入造成的假阳性和假阴性。



## **24. In-Context Defense in Computer Agents: An Empirical Study**

计算机代理中的上下文防御：实证研究 cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.

摘要: 由视觉语言模型(VLM)驱动的计算机代理极大地促进了人机交互，使用户能够通过自然语言指令执行复杂的任务。然而，这些代理容易受到上下文欺骗攻击，这是一种新兴的威胁，即对手在代理的操作环境中嵌入误导性内容，例如包含欺骗性指令的弹出窗口。现有的防御措施，如指示特工忽略欺骗性因素，已被证明在很大程度上无效。作为第一个关于保护计算机代理的系统研究，我们引入了Textbf[上下文中的防御}，利用上下文中的学习和思想链(COT)推理来对抗此类攻击。我们的方法涉及用一小组精心挑选的样本来增强代理的上下文，这些样本既包含恶意环境，也包含相应的防御响应。这些样本指导代理在行动计划之前首先执行明确的防御推理，降低对欺骗性攻击的敏感性。实验证明了该方法的有效性，对弹出窗口攻击的攻击成功率降低了91.2%，对环境注入攻击的平均成功率降低了74.6%，同时对分散注意力的广告实现了100%的成功防御。我们的发现强调：(1)为了获得最佳性能，防御推理必须先于行动计划，(2)最少的样本数量(少于3个)足以诱导代理人的防御行为。



## **25. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **26. AdvAD: Exploring Non-Parametric Diffusion for Imperceptible Adversarial Attacks**

AdvAD：探索不可感知的对抗攻击的非参数扩散 cs.LG

Accept by NeurIPS 2024. Please cite this paper using the following  format: J. Li, Z. He, A. Luo, J. Hu, Z. Wang, X. Kang*, "AdvAD: Exploring  Non-Parametric Diffusion for Imperceptible Adversarial Attacks", the 38th  Annual Conference on Neural Information Processing Systems (NeurIPS),  Vancouver, Canada, Dec 9-15, 2024. Code: https://github.com/XianguiKang/AdvAD

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09124v1) [paper-pdf](http://arxiv.org/pdf/2503.09124v1)

**Authors**: Jin Li, Ziqiang He, Anwei Luo, Jian-Fang Hu, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks aim to fool DNNs by adding imperceptible perturbation to the input data. Previous methods typically improve the imperceptibility of attacks by integrating common attack paradigms with specifically designed perception-based losses or the capabilities of generative models. In this paper, we propose Adversarial Attacks in Diffusion (AdvAD), a novel modeling framework distinct from existing attack paradigms. AdvAD innovatively conceptualizes attacking as a non-parametric diffusion process by theoretically exploring basic modeling approach rather than using the denoising or generation abilities of regular diffusion models requiring neural networks. At each step, much subtler yet effective adversarial guidance is crafted using only the attacked model without any additional network, which gradually leads the end of diffusion process from the original image to a desired imperceptible adversarial example. Grounded in a solid theoretical foundation of the proposed non-parametric diffusion process, AdvAD achieves high attack efficacy and imperceptibility with intrinsically lower overall perturbation strength. Additionally, an enhanced version AdvAD-X is proposed to evaluate the extreme of our novel framework under an ideal scenario. Extensive experiments demonstrate the effectiveness of the proposed AdvAD and AdvAD-X. Compared with state-of-the-art imperceptible attacks, AdvAD achieves an average of 99.9$\%$ (+17.3$\%$) ASR with 1.34 (-0.97) $l_2$ distance, 49.74 (+4.76) PSNR and 0.9971 (+0.0043) SSIM against four prevalent DNNs with three different architectures on the ImageNet-compatible dataset. Code is available at https://github.com/XianguiKang/AdvAD.

摘要: 不可察觉的敌意攻击旨在通过向输入数据添加不可察觉的扰动来愚弄DNN。以前的方法通常通过将常见的攻击范例与专门设计的基于感知的损失或生成模型的能力相结合来提高攻击的隐蔽性。本文提出了一种不同于现有攻击范式的建模框架--扩散中的对抗性攻击(AdvAD)。AdvAD创新性地将攻击定义为一个非参数扩散过程，从理论上探索了基本的建模方法，而不是利用需要神经网络的常规扩散模型的去噪或生成能力。在每个步骤中，只使用被攻击的模型而不需要任何额外的网络来构造更加微妙而有效的对抗指导，从而逐渐结束从原始图像到期望的不可感知的对抗示例的扩散过程。基于所提出的非参数扩散过程的坚实的理论基础，AdvAD以本质上较低的整体扰动强度实现了高攻击效率和不可感知性。此外，我们还提出了一个增强版本AdvAD-X来评估我们的新框架在理想场景下的极致。大量实验证明了所提出的AdvAD和AdvAD-X算法的有效性。与最新的不可察觉攻击相比，在与ImageNet兼容的数据集上，对于四种流行的DNN，AdvAD以1.3 4(-0.97)$L 2$距离、49.74(+4.76)PSNR和0.9971(+0.0043)SSIM获得了平均99.9$(+17.3$)ASR。代码可在https://github.com/XianguiKang/AdvAD.上找到



## **27. C^2 ATTACK: Towards Representation Backdoor on CLIP via Concept Confusion**

C2 ATTACK：通过概念混乱走向CLIP上的代表后门 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09095v1) [paper-pdf](http://arxiv.org/pdf/2503.09095v1)

**Authors**: Lijie Hu, Junchi Liao, Weimin Lyu, Shaopeng Fu, Tianhao Huang, Shu Yang, Guimin Hu, Di Wang

**Abstract**: Backdoor attacks pose a significant threat to deep learning models, enabling adversaries to embed hidden triggers that manipulate the behavior of the model during inference. Traditional backdoor attacks typically rely on inserting explicit triggers (e.g., external patches, or perturbations) into input data, but they often struggle to evade existing defense mechanisms. To address this limitation, we investigate backdoor attacks through the lens of the reasoning process in deep learning systems, drawing insights from interpretable AI. We conceptualize backdoor activation as the manipulation of learned concepts within the model's latent representations. Thus, existing attacks can be seen as implicit manipulations of these activated concepts during inference. This raises interesting questions: why not manipulate the concepts explicitly? This idea leads to our novel backdoor attack framework, Concept Confusion Attack (C^2 ATTACK), which leverages internal concepts in the model's reasoning as "triggers" without introducing explicit external modifications. By avoiding the use of real triggers and directly activating or deactivating specific concepts in latent spaces, our approach enhances stealth, making detection by existing defenses significantly harder. Using CLIP as a case study, experimental results demonstrate the effectiveness of C^2 ATTACK, achieving high attack success rates while maintaining robustness against advanced defenses.

摘要: 后门攻击对深度学习模型构成了重大威胁，使攻击者能够嵌入隐藏的触发器，在推理过程中操纵模型的行为。传统的后门攻击通常依赖于在输入数据中插入显式触发器(例如，外部补丁或扰动)，但它们往往难以避开现有的防御机制。为了解决这一局限性，我们通过深度学习系统中推理过程的镜头来调查后门攻击，从可解释的人工智能中获得见解。我们将后门激活概念化为在模型的潜在表示中操纵已学习的概念。因此，现有的攻击可以被视为在推理过程中对这些激活概念的隐式操纵。这引发了一些有趣的问题：为什么不明确地操纵这些概念呢？这个想法导致了我们的新的后门攻击框架，概念混淆攻击(C^2攻击)，它利用模型推理中的内部概念作为“触发器”，而不引入显式的外部修改。通过避免使用真实的触发器，并直接激活或停用潜在空间中的特定概念，我们的方法增强了隐蔽性，使现有防御系统的检测变得更加困难。以CLIP为例，实验结果证明了C^2攻击的有效性，在保持对高级防御的健壮性的同时，获得了较高的攻击成功率。



## **28. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但它们仍然容易受到对手操纵，例如通过即时注入攻击越狱。这些攻击绕过安全机制，生成受限或有害的内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全状态和越狱状态的潜在潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM的激活稳定在半稳定状态，可以被识别和扰动以诱导状态转变。使用降维技术，我们从安全和越狱反应中投影激活，以揭示低维空间中的潜在子空间。然后我们推导出一个扰动向量，当应用于安全表示时，将模型转变为越狱状态。我们的结果表明，这种因果干预导致在提示的子集中有统计意义的越狱反应。接下来，我们探索这些扰动是如何通过模型的各层传播的，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的发现表明，有针对性的扰动导致激活和模型反应的明显转变。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转变为先发制人的、模型不可知的技术，在表征水平上中和敌对状态。



## **29. Quantitative Analysis of Deeply Quantized Tiny Neural Networks Robust to Adversarial Attacks**

对对抗攻击鲁棒的深度量化微型神经网络的定量分析 cs.LG

arXiv admin note: substantial text overlap with arXiv:2304.12829

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08973v1) [paper-pdf](http://arxiv.org/pdf/2503.08973v1)

**Authors**: Idris Zakariyya, Ferheen Ayaz, Mounia Kharbouche-Harrari, Jeremy Singer, Sye Loong Keoh, Danilo Pau, José Cano

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, especially Deep Neural Networks (DNNs), is imperative to facilitate their deployment on resource-constrained edge devices. However, a notable drawback of DNN models lies in their susceptibility to adversarial attacks, wherein minor input perturbations can deceive them. A primary challenge revolves around the development of accurate, resilient, and compact DNN models suitable for deployment on resource-constrained edge devices. This paper presents the outcomes of a compact DNN model that exhibits resilience against both black-box and white-box adversarial attacks. This work has achieved this resilience through training with the QKeras quantization-aware training framework. The study explores the potential of QKeras and an adversarial robustness technique, Jacobian Regularization (JR), to co-optimize the DNN architecture through per-layer JR methodology. As a result, this paper has devised a DNN model employing this co-optimization strategy based on Stochastic Ternary Quantization (STQ). Its performance was compared against existing DNN models in the face of various white-box and black-box attacks. The experimental findings revealed that, the proposed DNN model had small footprint and on average, it exhibited better performance than Quanos and DS-CNN MLCommons/TinyML (MLC/T) benchmarks when challenged with white-box and black-box attacks, respectively, on the CIFAR-10 image and Google Speech Commands audio datasets.

摘要: 减少机器学习(ML)模型的存储空间，特别是深度神经网络(DNN)模型，对于促进它们在资源受限的边缘设备上的部署是必不可少的。然而，DNN模型的一个显著缺陷在于它们对对抗性攻击的敏感性，其中微小的输入扰动可以欺骗它们。主要的挑战围绕着开发适合在资源受限的边缘设备上部署的准确、有弹性且紧凑的DNN模型。本文给出了一个紧凑的DNN模型的结果，该模型对黑盒和白盒对手攻击都表现出了韧性。这项工作通过使用QKera量化感知培训框架进行培训，实现了这种复原力。这项研究探索了QKERA和一种对抗性健壮性技术--雅可比正则化(JR)--通过每层JR方法共同优化DNN结构的潜力。因此，本文设计了一个基于随机三进制量化(STQ)的DNN模型，该模型采用了这种联合优化策略。在面对各种白盒和黑盒攻击时，将其性能与现有的DNN模型进行了比较。实验结果表明，所提出的DNN模型占用空间小，在CIFAR-10图像和Google语音命令音频数据集上分别受到白盒和黑盒攻击时，其性能平均优于Quanos和DS-CNN MLCommons/TinyML(MLC/T)基准测试。



## **30. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它对闭源模型表现出很强的攻击可转移性，尽管只在Llama3上进行了优化，但在GPT-3.5上达到了99 ASR，在GPT-4上达到了49 ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



## **31. Backtracking for Safety**

为了安全而回溯 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但确保它们的安全性和与人类价值观的一致性仍然至关重要。当前的安全对齐方法，如监督微调和基于强化学习的方法，可能会表现出对对手攻击的脆弱性，并且往往导致浅层安全对齐，主要侧重于防止生成输出的初始令牌中的有害内容。虽然像重置这样的方法可以通过丢弃之前的令牌并重新启动生成过程来帮助从不安全的世代中恢复，但它们不太适合于解决细微差别的安全违规行为，如毒性，否则可能会在良性和漫长的世代中出现。在本文中，我们提出了一种新的回溯方法，旨在解决这些局限性。我们的方法允许模型在发电过程中发生安全违规时恢复到更安全的发电状态，而不一定是在开始时。这种方法能够在不丢弃整个生成的文本的情况下有针对性地纠正有问题的片段，从而保持效率。我们证明，我们的方法大大减少了在产生过程中出现的毒性，对效率的影响最小。



## **32. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

人类可读的对抗性预言：使用情境背景对LLM漏洞的调查 cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.

摘要: 之前发现大型语言模型（LLM）漏洞的研究经常使用无意义的对抗提示。然而，现在可以使用自动检测技术轻松识别此类提示。为了进一步加强对抗性攻击，我们重点关注人类可读的对抗性提示，这些提示是更现实、更强大的威胁。我们的主要贡献是（1）情境驱动攻击利用电影剧本作为上下文来创建人类可读的提示，成功欺骗LLM，（2）对抗性后缀转换，将无意义的对抗性后缀转换为独立有意义的文本，以及（3）具有p核采样的Advancer，一种生成多样化、人类可读的对抗性后缀的方法，提高GPT-3.5和Gemma 7 B等模型的攻击功效。



## **33. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

鸟看起来像汽车：本质上可解释的深度学习的对抗分析 cs.LG

Preprint

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08636v1) [paper-pdf](http://arxiv.org/pdf/2503.08636v1)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of prototype-based networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.

摘要: 人们普遍认为，本质上可解释的深度学习模型确保了对其行为的正确、直观的理解，并提供了针对意外错误或故意操作的更强的健壮性。然而，这些信念还没有得到全面证实，越来越多的证据让人对它们产生了怀疑。在这篇文章中，我们强调了与过度依赖和易受敌意操纵有关的风险，这些模型被设计为“本质上(也就是内在)可解释的”模型。我们介绍了利用原型操纵和后门攻击对基于原型的网络进行敌意分析的两种策略，并讨论了概念瓶颈模型如何防御这些攻击。通过利用潜在原型来愚弄模型的推理，表明了深层神经网络固有的不可解释性，导致了一种错误的安全感，并被视觉确认偏差所强化。已报道的基于原型的网络的局限性使它们的可信性和适用性受到质疑，促使进一步研究(深度)可解释模型的健壮性和一致性。



## **34. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2501.06044v4) [paper-pdf](http://arxiv.org/pdf/2501.06044v4)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议可实现的最佳容错已在广泛的设置中得到了表征。例如，对于在部分同步设置中操作的状态机复制(SMR)协议，当且仅当$\Alpha+2\Beta\leq 1$时，可以同时保证针对$\Alpha$受限的对手(即，控制少于$\Alpha$部分参与者的对手)的一致性和针对$\beta$受限的攻击者的活性。本文刻画了当标准一致性要求被放宽以允许有限数量的一致性违规时，SMR协议在多大程度上可能获得比最优更好的容错保证。我们证明了如果没有额外的时间假设，绑定回滚是不可能的，并研究了当攻击时间附近的消息延迟由参数$\Delta^*$(该参数可以任意大于在部分同步模型中限制GST后消息延迟的参数$\Delta$)限定时，容忍一致性违规并从一致性违规中恢复的协议。这里，协议的容错性可以是$r$的非常数函数，我们证明了对于每个$r$，协议的最优“可恢复容错性”的上下界是匹配的。例如，对于在部分同步设置中保证对1/3有界攻击者的活跃性的协议，5/9有界的攻击者总是可以导致一次一致性违反而不是两次，而2/3有界的攻击者总是可以引起两次一致性违反而不是三次一致性违反。我们的积极结果是通过通用的“恢复程序”实现的，该程序可以嫁接到任何负责任的SMR协议上，并在违规后恢复一致性，同时仅回滚在前$2\Delta^*$时间步长中完成的事务。



## **35. Low-Cost Privacy-Preserving Decentralized Learning**

低成本保护隐私的分散学习 cs.LG

24 pages, accepted at Pets 2025

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2403.11795v3) [paper-pdf](http://arxiv.org/pdf/2403.11795v3)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: Decentralized learning (DL) is an emerging paradigm of collaborative machine learning that enables nodes in a network to train models collectively without sharing their raw data or relying on a central server. This paper introduces Zip-DL, a privacy-aware DL algorithm that leverages correlated noise to achieve robust privacy against local adversaries while ensuring efficient convergence at low communication costs. By progressively neutralizing the noise added during distributed averaging, Zip-DL combines strong privacy guarantees with high model accuracy. Its design requires only one communication round per gradient descent iteration, significantly reducing communication overhead compared to competitors. We establish theoretical bounds on both convergence speed and privacy guarantees. Moreover, extensive experiments demonstrating Zip-DL's practical applicability make it outperform state-of-the-art methods in the accuracy vs. vulnerability trade-off. Specifically, Zip-DL (i) reduces membership-inference attack success rates by up to 35% compared to baseline DL, (ii) decreases attack efficacy by up to 13% compared to competitors offering similar utility, and (iii) achieves up to 59% higher accuracy to completely nullify a basic attack scenario, compared to a state-of-the-art privacy-preserving approach under the same threat model. These results position Zip-DL as a practical and efficient solution for privacy-preserving decentralized learning in real-world applications.

摘要: 分散学习是一种新兴的协作机器学习范式，它使网络中的节点能够在不共享原始数据或依赖中央服务器的情况下集体训练模型。本文介绍了ZIP-DL算法，这是一种隐私感知的DL算法，它利用相关噪声来实现对本地攻击者的稳健隐私保护，同时确保以较低的通信成本实现高效的收敛。通过逐步中和分布式平均过程中添加的噪声，Zip-DL将强大的隐私保证与高模型精度相结合。它的设计在每次梯度下降迭代中只需要一轮通信，与竞争对手相比显著减少了通信开销。我们建立了收敛速度和隐私保障的理论界限。此外，大量实验证明了Zip-DL的实用适用性，使其在准确性与脆弱性之间的权衡上优于最先进的方法。具体地说，与基准DL相比，Zip-DL(I)将成员资格推理攻击成功率降低了高达35%，(Ii)与提供类似实用工具的竞争对手相比，攻击效率降低了高达13%，(Iii)与相同威胁模型下最先进的隐私保护方法相比，可实现高达59%的准确率，以完全消除基本攻击场景。这些结果将Zip-DL定位为现实世界应用程序中保护隐私的分散学习的实用而有效的解决方案。



## **36. Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**

Adv-CPG：具有面部对抗攻击的定制肖像生成框架 cs.CV

Accepted by CVPR-25

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08269v1) [paper-pdf](http://arxiv.org/pdf/2503.08269v1)

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively.

摘要: 最近的定制肖像生成(CPG)方法以面部图像和文本提示为输入，引起了广泛的关注。虽然这些方法会生成高保真的肖像，但它们无法防止生成的肖像被恶意的人脸识别系统跟踪和滥用。针对这一问题，提出了一种基于人脸对抗攻击的个性化肖像生成框架(ADV-CPG)。具体地说，为了实现面部隐私保护，我们设计了一个轻量级的本地ID加密器和一个加密增强器。它们分别通过直接注入目标身份和添加额外的身份指导来实施渐进式双层加密保护。此外，为了实现细粒度和个性化的肖像生成，我们开发了一个能够生成受控细粒度人脸特征的多模式图像定制器。据我们所知，ADV-CPG是第一个将面部对抗性攻击引入CPG的研究。实验结果表明，ADV-CPG的平均攻击成功率比基于噪声的SOTA攻击方法和无约束攻击方法分别高出28.1%和2.86%。



## **37. A Grey-box Text Attack Framework using Explainable AI**

使用可解释人工智能的灰箱文本攻击框架 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08226v1) [paper-pdf](http://arxiv.org/pdf/2503.08226v1)

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models.

摘要: 可解释人工智能是一种强大的策略，用于理解人类可解释语言中的复杂黑盒模型预测。它提供了使用值得信赖和可靠的人工智能系统所需的证据。然而，另一方面，它也为定位人工智能模型中可能的漏洞打开了大门。传统的对抗性文本攻击使用单词替换、数据增强技术和基于梯度的攻击，对来自Transformers(BERT)变体的强大的预训练双向编码器表示进行攻击，以生成对抗性句子。这些攻击通常是白盒性质的，并不实用，因为它们很容易被人类发现，例如，将单词从“穷”改为“富”。我们提出了一种简单而有效的灰盒和黑盒方法，该方法不需要模型知识，同时使用一组代理Transformer/BERT模型来执行攻击，使用可解释的人工智能技术。由于Transformers是几乎所有自然语言处理(NLP)任务的当前最先进的模型，从BERT1生成的攻击可以转移到BERT2。这种可转移性之所以成为可能，是因为转换器中的注意力机制允许模型捕获序列中的远程依赖关系。使用通过注意的BERT泛化的力量，我们试图通过攻击几个都基于不同体系结构的代理变压器变体来探索变压器是如何学习的。我们证明了这种方法是非常有效的，可以通过更改一个人类无法检测的单词来生成语义良好的句子，同时仍然可以愚弄其他BERT模型。



## **38. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **39. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

对话注入攻击：通过上下文操纵越狱LLM cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.

摘要: 大型语言模型(LLM)在广泛的应用程序中显示出了重要的实用价值；然而，它们的部署受到安全漏洞的困扰，特别是越狱攻击。这些攻击通过精心编制敌意提示来操纵LLM生成有害或不道德的内容。虽然目前对越狱攻击的大部分研究都集中在单回合互动上，但在很大程度上忽视了历史对话对模型行为的影响。在本文中，我们介绍了一种新的越狱范例，对话注入攻击(DIA)，它利用对话历史来提高此类攻击的成功率。DIA在黑盒设置中运行，只需要访问聊天API或了解LLM的聊天模板。我们提出了两种构造对抗性历史对话的方法：一种是采用灰盒预填充攻击，另一种是利用延迟响应。我们的实验表明，DIA在包括Llama-3.1和GPT-40在内的最近的LLM上达到了最先进的攻击成功率。此外，我们还证明了DIA可以绕过5种不同的防御机制，突出了其健壮性和有效性。



## **40. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

MAGIC：通过协作LLM代理掌握上下文中的物理对抗生成 cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.08014v2) [paper-pdf](http://arxiv.org/pdf/2412.08014v2)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world environments and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate deployment within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and generates adversarial patch through the synergistic interaction of language and vision capabilities. In particular, MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal deployment strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical levels, i.e., nuImage and manually captured real-world scenes, where both statistical and visual results prove that our MAGIC is powerful and effective for attacking widely applied object detection systems, i.e., YOLO and DETR series.

摘要: 驾驶场景中的物理对抗性攻击可以暴露视觉感知模型中的关键漏洞。然而，由于现实世界环境的多样性和保持视觉自然性的要求，开发此类攻击仍然具有挑战性。在这一挑战的基础上，我们将物理对抗性攻击重新定义为一次性补丁生成问题。我们的方法通过深度生成模型生成对抗性补丁，该模型考虑了特定的场景上下文，支持在匹配环境中直接物理部署。主要挑战在于同时实现两个目标：生成有效误导目标检测系统的对抗性补丁，同时确定场景中的上下文适当部署。我们提出了MAGIC(掌握上下文中的物理对手生成)，这是一个由多模式LLM代理支持的新框架来应对这些挑战。Magic自动理解场景背景，并通过语言和视觉能力的协同交互生成对抗性补丁。特别是，Magic协调了三个专门的LLM代理：adv-patch生成代理(Gagent)通过针对文本到图像模型的战略提示工程掌握了欺骗性补丁的创建。Adv-patch部署代理通过基于场景理解来确定最佳部署策略，从而确保上下文的一致性。自我检查代理(EAgent)通过提供对两个过程的关键监督和迭代细化来完成这三部曲。我们在数字和物理两个层面上验证了我们的方法，即NuImage和手动捕获的真实场景，其中统计和视觉结果都证明了我们的魔力对于攻击广泛应用的目标检测系统(如YOLO和DETR系列)是强大和有效的。



## **41. MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation**

MIGA：对语义操纵去噪模型的相互信息引导攻击 cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.06966v2) [paper-pdf](http://arxiv.org/pdf/2503.06966v2)

**Authors**: Guanghao Li, Mingzhi Chen, Hao Yu, Shuting Dong, Wenhao Jiang, Ming Tang, Chun Yuan

**Abstract**: Deep learning-based denoising models have been widely employed in vision tasks, functioning as filters to eliminate noise while retaining crucial semantic information. Additionally, they play a vital role in defending against adversarial perturbations that threaten downstream tasks. However, these models can be intrinsically susceptible to adversarial attacks due to their dependence on specific noise assumptions. Existing attacks on denoising models mainly aim at deteriorating visual clarity while neglecting semantic manipulation, rendering them either easily detectable or limited in effectiveness. In this paper, we propose Mutual Information-Guided Attack (MIGA), the first method designed to directly attack deep denoising models by strategically disrupting their ability to preserve semantic content via adversarial perturbations. By minimizing the mutual information between the original and denoised images, a measure of semantic similarity. MIGA forces the denoiser to produce perceptually clean yet semantically altered outputs. While these images appear visually plausible, they encode systematically distorted semantics, revealing a fundamental vulnerability in denoising models. These distortions persist in denoised outputs and can be quantitatively assessed through downstream task performance. We propose new evaluation metrics and systematically assess MIGA on four denoising models across five datasets, demonstrating its consistent effectiveness in disrupting semantic fidelity. Our findings suggest that denoising models are not always robust and can introduce security risks in real-world applications.

摘要: 基于深度学习的去噪模型被广泛应用于视觉任务中，在保留关键语义信息的同时起到滤除噪声的作用。此外，它们在防御威胁下游任务的对抗性干扰方面发挥着至关重要的作用。然而，由于这些模型依赖于特定的噪声假设，因此可能本质上容易受到对抗性攻击。现有的对去噪模型的攻击主要是针对视觉清晰度的下降，而忽略了语义操作，使得它们要么容易被检测到，要么效果有限。在本文中，我们提出了互信息制导攻击(MIGA)，这是第一种设计用于直接攻击深度去噪模型的方法，该方法通过对抗性扰动来战略性地破坏深度去噪模型保持语义内容的能力。通过最小化原始图像和去噪图像之间的互信息，来衡量语义相似性。MIGA强制去噪器产生感知上干净但语义改变的输出。虽然这些图像在视觉上看起来是可信的，但它们编码的是系统性扭曲的语义，揭示了去噪模型中的一个根本漏洞。这些失真持续存在于去噪输出中，并可通过下游任务执行情况进行量化评估。我们提出了新的评价指标，并在五个数据集的四个去噪模型上对MIGA进行了系统的评估，证明了它在破坏语义保真度方面的一致有效性。我们的发现表明，去噪模型并不总是健壮的，可能会在现实世界的应用中引入安全风险。



## **42. Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks**

通过更强的个人攻击实现百万规模的对抗稳健性评估 cs.LG

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2411.15210v4) [paper-pdf](http://arxiv.org/pdf/2411.15210v4)

**Authors**: Yong Xie, Weijie Zheng, Hanxun Huang, Guangnan Ye, Xingjun Ma

**Abstract**: As deep learning models are increasingly deployed in safety-critical applications, evaluating their vulnerabilities to adversarial perturbations is essential for ensuring their reliability and trustworthiness. Over the past decade, a large number of white-box adversarial robustness evaluation methods (i.e., attacks) have been proposed, ranging from single-step to multi-step methods and from individual to ensemble methods. Despite these advances, challenges remain in conducting meaningful and comprehensive robustness evaluations, particularly when it comes to large-scale testing and ensuring evaluations reflect real-world adversarial risks. In this work, we focus on image classification models and propose a novel individual attack method, Probability Margin Attack (PMA), which defines the adversarial margin in the probability space rather than the logits space. We analyze the relationship between PMA and existing cross-entropy or logits-margin-based attacks, and show that PMA can outperform the current state-of-the-art individual methods. Building on PMA, we propose two types of ensemble attacks that balance effectiveness and efficiency. Furthermore, we create a million-scale dataset, CC1M, derived from the existing CC3M dataset, and use it to conduct the first million-scale white-box adversarial robustness evaluation of adversarially-trained ImageNet models. Our findings provide valuable insights into the robustness gaps between individual versus ensemble attacks and small-scale versus million-scale evaluations.

摘要: 随着深度学习模型越来越多地被部署在安全关键应用中，评估它们对敌意扰动的脆弱性对于确保它们的可靠性和可信性至关重要。在过去的十年里，已经提出了大量的白盒对抗健壮性评估方法(即攻击)，从单步方法到多步方法，从个体方法到集成方法。尽管取得了这些进展，但在进行有意义和全面的稳健性评估方面仍然存在挑战，特别是在进行大规模测试和确保评估反映现实世界的对抗性风险方面。本文重点研究了图像分类模型，提出了一种新的个体攻击方法--概率边缘攻击(PMA)，它在概率空间而不是Logits空间中定义了敌方边缘。我们分析了PMA与现有的基于交叉熵或Logits差值的攻击之间的关系，并证明了PMA的性能优于目前最先进的个别方法。在PMA的基础上，我们提出了两种平衡有效性和效率的集成攻击。此外，我们从现有的CC3M数据集中创建了一个百万尺度的数据集CC1M，并使用它对经过对手训练的ImageNet模型进行了第一次百万尺度的白盒对抗健壮性评估。我们的发现对个体攻击与整体攻击以及小规模评估与百万规模评估之间的稳健性差距提供了有价值的见解。



## **43. FairDeFace: Evaluating the Fairness and Adversarial Robustness of Face Obfuscation Methods**

FairDeFace：评估面部模糊方法的公平性和对抗鲁棒性 cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08731v1) [paper-pdf](http://arxiv.org/pdf/2503.08731v1)

**Authors**: Seyyed Mohammad Sadegh Moosavi Khorzooghi, Poojitha Thota, Mohit Singhal, Abolfazl Asudeh, Gautam Das, Shirin Nilizadeh

**Abstract**: The lack of a common platform and benchmark datasets for evaluating face obfuscation methods has been a challenge, with every method being tested using arbitrary experiments, datasets, and metrics. While prior work has demonstrated that face recognition systems exhibit bias against some demographic groups, there exists a substantial gap in our understanding regarding the fairness of face obfuscation methods. Providing fair face obfuscation methods can ensure equitable protection across diverse demographic groups, especially since they can be used to preserve the privacy of vulnerable populations. To address these gaps, this paper introduces a comprehensive framework, named FairDeFace, designed to assess the adversarial robustness and fairness of face obfuscation methods. The framework introduces a set of modules encompassing data benchmarks, face detection and recognition algorithms, adversarial models, utility detection models, and fairness metrics. FairDeFace serves as a versatile platform where any face obfuscation method can be integrated, allowing for rigorous testing and comparison with other state-of-the-art methods. In its current implementation, FairDeFace incorporates 6 attacks, and several privacy, utility and fairness metrics. Using FairDeFace, and by conducting more than 500 experiments, we evaluated and compared the adversarial robustness of seven face obfuscation methods. This extensive analysis led to many interesting findings both in terms of the degree of robustness of existing methods and their biases against some gender or racial groups. FairDeFace also uses visualization of focused areas for both obfuscation and verification attacks to show not only which areas are mostly changed in the obfuscation process for some demographics, but also why they failed through focus area comparison of obfuscation and verification.

摘要: 缺乏用于评估人脸模糊方法的通用平台和基准数据集一直是一个挑战，每种方法都使用任意的实验、数据集和度量进行测试。虽然先前的工作表明，人脸识别系统表现出对某些人口群体的偏见，但我们对人脸混淆方法的公平性的理解存在着很大的差距。提供公平的人脸模糊方法可以确保对不同人口群体的公平保护，特别是因为这些方法可以用来保护弱势群体的隐私。为了解决这些不足，本文引入了一个名为FairDeFace的综合框架，旨在评估人脸混淆方法的对抗健壮性和公平性。该框架引入了一系列模块，包括数据基准、人脸检测和识别算法、对抗性模型、效用检测模型和公平度量。FairDeFace是一个多功能的平台，可以集成任何人脸模糊方法，允许进行严格的测试并与其他最先进的方法进行比较。在目前的实现中，FairDeFace包含6个攻击，以及几个隐私、效用和公平性指标。使用FairDeFace，通过500多个实验，评估和比较了七种人脸混淆方法的对抗健壮性。这种广泛的分析在现有方法的稳健性程度及其对某些性别或种族群体的偏见方面导致了许多有趣的发现。FairDeFace还使用针对混淆和验证攻击的焦点区域可视化，不仅可以显示某些人口的混淆过程中哪些区域变化最大，还可以通过混淆和验证的焦点区域比较来显示它们失败的原因。



## **44. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.12478v2) [paper-pdf](http://arxiv.org/pdf/2412.12478v2)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.

摘要: 基于DNN的语言模型在各种任务中表现出色，但即使是Sota LLM也容易受到文本攻击。对抗性语篇在自然语言处理的多个子领域发挥着至关重要的作用。然而，目前的研究存在以下问题。(1)大多数文本对抗性攻击方法针对的是资源丰富的语言。如何为较少研究的语言生成对抗性文本？(2)大多数文本对抗性攻击方法容易产生无效或歧义的对抗性文本。我们如何构建高质量的对抗性健壮性基准？(3)新的语言模型可能对先前生成的部分对抗性文本免疫。我们如何更新对手健壮性基准？为了解决上述问题，我们引入了HITL-GAT，这是一个基于人在环中生成对抗性文本的通用方法的系统。HITL-GAT在一条流水线上包括四个阶段：受害者模型构建、对手实例生成、高质量基准构建和对手健壮性评估。此外，我们还利用HITL-GAT对藏文进行了实例研究，对其他研究较少的语言的对抗性研究具有一定的借鉴意义。



## **45. Safety Guardrails for LLM-Enabled Robots**

LLM支持机器人的安全护栏 cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.

摘要: 尽管将大型语言模型(LLM)集成到机器人学中释放了变革性的能力，但它也带来了重大的安全问题，从平均情况下的LLM错误(例如，幻觉)到对抗性的越狱攻击，这可能会在现实世界中产生有害的机器人行为。传统的机器人安全方法不能解决LLM的新脆弱性，而当前的LLM安全护栏忽略了机器人在动态真实环境中操作所带来的物理风险。在本文中，我们提出了一种两级护栏结构RoboGuard，以确保LLM支持的机器人的安全。RoboGuard首先通过使用信任根LLM将预定义的安全规则与机器人环境相关联，该LLM使用思想链(COT)推理来生成严格的安全规范，如时间逻辑约束。然后，RoboGuard使用时态逻辑控制合成来解决这些上下文安全规范和可能不安全的计划之间的潜在冲突，这确保了安全合规性，同时将违反用户偏好的程度降至最低。通过考虑最坏情况下的越狱攻击的大量模拟和真实世界实验，我们证明了RoboGuard在不影响安全计划的性能的情况下，将不安全计划的执行率从92%降低到2.5%以下。我们还证明了RoboGuard是资源高效的，对自适应攻击具有健壮性，并且通过使其信任根LLM能够执行CoT推理而显著增强。这些结果突显了RoboGuard在缓解安全风险和提高启用LLM的机器人可靠性方面的潜力。



## **46. ReLATE: Resilient Learner Selection for Multivariate Time-Series Classification Against Adversarial Attacks**

ReLATE：针对对抗性攻击的多元时间序列分类的弹性学习者选择 cs.LG

Accepted by the AAAI-25 Workshop on Artificial Intelligence for Time  Series Analysis (AI4TS)

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07882v1) [paper-pdf](http://arxiv.org/pdf/2503.07882v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Aaron Tartz, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. We introduce ReLATE, a framework that identifies robust learners based on dataset similarity, reduces computational overhead, and enhances resilience. ReLATE maintains multiple deep learning models in well-known adversarial attack scenarios, capturing model performance. ReLATE identifies the most analogous dataset to a given target using a similarity metric, then applies the optimal model from the most similar dataset. ReLATE reduces computational overhead by an average of 81.2%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 4.2% of Oracle.

摘要: 最大限度地减少时间序列分类中的计算负担，特别是深度学习模型中的计算负担，提出了一个重大挑战。对抗性攻击进一步加剧了这一挑战，强调了对确保稳健性能和高效模型选择的弹性方法的需要。我们引入了ReLATE，这是一个基于数据集相似性识别稳健学习器、减少计算负担并增强弹性的框架。ReLATE在众所周知的对抗性攻击场景中维护多个深度学习模型，捕捉模型性能。ReLATE使用相似性指标识别与给定目标最相似的数据集，然后应用来自最相似数据集的最佳模型。ReLATE平均降低了81.2%的计算负担，增强了对抗弹性并简化了稳健的模型选择，而所有这些都不会牺牲性能，仅为Oracle的4.2%。



## **47. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

论多数票签名新加坡元的拜占庭式过失容忍 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2502.19170v2) [paper-pdf](http://arxiv.org/pdf/2502.19170v2)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.

摘要: 在分布式学习中，基于符号的压缩算法，如多数投票的signSGD，提供了一种轻量级的SGD替代方案，具有额外的优势：几乎是免费的容错。然而，对于拥有多数票的signSGD来说，这种容错已经被证明只涵盖较弱的对手的情况，即那些不是无所不知的或不能串通以基于常识和策略的攻击的情况。在这项工作中，我们缩小了这一差距，并提供了新的见解，即拥有多数选票的signSGD如何具有抵御无所不知和串通的对手的能力，这些对手在与其他对手沟通后策划攻击，从而拥有更好的信息，基于共同的最优策略执行最具破坏性的攻击。我们的核心贡献是提供了一种证明，首先定义了无所不知的框架，并以多数票对signSGD造成了最强的破坏，而不对攻击者施加任何限制。由于基于符号的方法的过滤效果，我们将攻击空间上界到最优策略，以最大化攻击者的损害。因此，我们在不求助于未知常量的情况下，得到了不正确聚集的显式概率界，在拜占庭攻击者存在的情况下，提供了带多数投票的signSGD的收敛界，并提供了精确的收敛速度。我们的发现得到了在MNIST数据集上的实验支持，该实验在分布式学习环境中具有不同强度的对手。



## **48. Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters**

使用性能计数器检测人工智能加速器中的对抗攻击 cs.CR

7 pages, 8 figures

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07568v1) [paper-pdf](http://arxiv.org/pdf/2503.07568v1)

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia

**Abstract**: Rapid adoption of AI technologies raises several major security concerns, including the risks of adversarial perturbations, which threaten the confidentiality and integrity of AI applications. Protecting AI hardware from misuse and diverse security threats is a challenging task. To address this challenge, we propose SAMURAI, a novel framework for safeguarding against malicious usage of AI hardware and its resilience to attacks. SAMURAI introduces an AI Performance Counter (APC) for tracking dynamic behavior of an AI model coupled with an on-chip Machine Learning (ML) analysis engine, known as TANTO (Trained Anomaly Inspection Through Trace Observation). APC records the runtime profile of the low-level hardware events of different AI operations. Subsequently, the summary information recorded by the APC is processed by TANTO to efficiently identify potential security breaches and ensure secure, responsible use of AI. SAMURAI enables real-time detection of security threats and misuse without relying on traditional software-based solutions that require model integration. Experimental results demonstrate that SAMURAI achieves up to 97% accuracy in detecting adversarial attacks with moderate overhead on various AI models, significantly outperforming conventional software-based approaches. It enhances security and regulatory compliance, providing a comprehensive solution for safeguarding AI against emergent threats.

摘要: 人工智能技术的快速采用带来了几个主要的安全问题，包括对抗性扰动的风险，这威胁到人工智能应用程序的机密性和完整性。保护人工智能硬件免受滥用和多样化的安全威胁是一项具有挑战性的任务。为了应对这一挑战，我们提出了武士，这是一个新的框架，用于防范恶意使用人工智能硬件及其对攻击的弹性。Samurai引入了AI性能计数器(APC)来跟踪AI模型的动态行为，并结合了芯片上的机器学习(ML)分析引擎，称为Tanto(通过跟踪观察进行训练的异常检测)。APC记录不同AI操作的底层硬件事件的运行时配置文件。随后，APC记录的摘要信息由Tanto处理，以有效地识别潜在的安全漏洞，并确保安全、负责任地使用人工智能。Samurai能够实时检测安全威胁和滥用，而不需要依赖需要模型集成的传统基于软件的解决方案。实验结果表明，武士在各种人工智能模型上以适度的开销检测敌意攻击的准确率高达97%，显著优于传统的基于软件的方法。它增强了安全性和监管合规性，为保护人工智能免受紧急威胁提供了全面的解决方案。



## **49. Transform-Dependent Adversarial Attacks**

依赖转换的对抗攻击 cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.08443v2) [paper-pdf](http://arxiv.org/pdf/2406.08443v2)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: Deep networks are highly vulnerable to adversarial attacks, yet conventional attack methods utilize static adversarial perturbations that induce fixed mispredictions. In this work, we exploit an overlooked property of adversarial perturbations--their dependence on image transforms--and introduce transform-dependent adversarial attacks. Unlike traditional attacks, our perturbations exhibit metamorphic properties, enabling diverse adversarial effects as a function of transformation parameters. We demonstrate that this transform-dependent vulnerability exists across different architectures (e.g., CNN and transformer), vision tasks (e.g., image classification and object detection), and a wide range of image transforms. Additionally, we show that transform-dependent perturbations can serve as a defense mechanism, preventing sensitive information disclosure when image enhancement transforms pose a risk of revealing private content. Through analysis in blackbox and defended model settings, we show that transform-dependent perturbations achieve high targeted attack success rates, outperforming state-of-the-art transfer attacks by 17-31% in blackbox scenarios. Our work introduces novel, controllable paradigm for adversarial attack deployment, revealing a previously overlooked vulnerability in deep networks.

摘要: 深层网络很容易受到敌意攻击，然而传统的攻击方法利用静态的对抗性扰动，导致固定的错误预测。在这项工作中，我们利用了对抗性扰动的一个被忽视的属性--它们对图像变换的依赖--并引入了依赖于变换的对抗性攻击。与传统的攻击不同，我们的扰动表现出变形的性质，使不同的对抗效果作为变换参数的函数。我们证明了这种依赖于变换的漏洞存在于不同的架构(例如，CNN和Transformer)、视觉任务(例如，图像分类和目标检测)以及广泛的图像变换中。此外，我们还证明了依赖于变换的扰动可以作为一种防御机制，当图像增强变换带来泄露隐私内容的风险时，可以防止敏感信息泄露。通过在黑盒和防御模型设置下的分析，我们证明了依赖于变换的扰动获得了高的目标攻击成功率，在黑盒场景中比最先进的传输攻击高出17%-31%。我们的工作为对抗性攻击部署引入了新的、可控的范例，揭示了以前在深层网络中被忽视的漏洞。



## **50. Learning to Localize Leakage of Cryptographic Sensitive Variables**

学习本地化加密敏感变量的泄漏 cs.LG

52 pages, 30 figures. Our code can be found at  https://github.com/jimgammell/learning_to_localize_leakage

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07464v1) [paper-pdf](http://arxiv.org/pdf/2503.07464v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, Kaushik Roy

**Abstract**: While cryptographic algorithms such as the ubiquitous Advanced Encryption Standard (AES) are secure, *physical implementations* of these algorithms in hardware inevitably 'leak' sensitive data such as cryptographic keys. A particularly insidious form of leakage arises from the fact that hardware consumes power and emits radiation in a manner that is statistically associated with the data it processes and the instructions it executes. Supervised deep learning has emerged as a state-of-the-art tool for carrying out *side-channel attacks*, which exploit this leakage by learning to map power/radiation measurements throughout encryption to the sensitive data operated on during that encryption. In this work we develop a principled deep learning framework for determining the relative leakage due to measurements recorded at different points in time, in order to inform *defense* against such attacks. This information is invaluable to cryptographic hardware designers for understanding *why* their hardware leaks and how they can mitigate it (e.g. by indicating the particular sections of code or electronic components which are responsible). Our framework is based on an adversarial game between a family of classifiers trained to estimate the conditional distributions of sensitive data given subsets of measurements, and a budget-constrained noise distribution which probabilistically erases individual measurements to maximize the loss of these classifiers. We demonstrate our method's efficacy and ability to overcome limitations of prior work through extensive experimental comparison with 8 baseline methods using 3 evaluation metrics and 6 publicly-available power/EM trace datasets from AES, ECC and RSA implementations. We provide an open-source PyTorch implementation of these experiments.

摘要: 虽然加密算法(如无处不在的高级加密标准(AES))是安全的，但这些算法在硬件中的*物理实现*不可避免地‘泄露’敏感数据，如密钥。一种特别隐蔽的泄漏形式源于这样一个事实，即硬件消耗功率并发出辐射，其方式与其处理的数据和执行的指令在统计上相关联。有监督的深度学习已经成为一种执行*侧通道攻击*的最先进工具，它通过学习将整个加密过程中的功率/辐射测量映射到加密过程中操作的敏感数据来利用这种泄漏。在这项工作中，我们开发了一个原则性的深度学习框架，用于确定由于在不同时间点记录的测量而导致的相对泄漏，以便为防御此类攻击提供信息。这些信息对于密码硬件设计者来说是非常有价值的，他们可以理解他们的硬件泄漏以及他们如何减轻泄漏(例如，通过指示负责的特定代码段或电子组件)。我们的框架基于一组分类器和预算受限噪声分布之间的对抗性博弈，前者被训练为估计给定测量子集的敏感数据的条件分布，后者概率地擦除单个测量以最大化这些分类器的损失。我们通过使用3个评估指标和6个来自AES、ECC和RSA实现的公开可用的POWER/EM跟踪数据集与8种基线方法进行广泛的实验比较，证明了我们的方法的有效性和克服先前工作的局限性的能力。我们提供了这些实验的开放源码的PyTorch实现。



