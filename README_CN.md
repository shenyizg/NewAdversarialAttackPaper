# Latest Adversarial Attack Papers
**update at 2025-04-21 09:58:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robust Decentralized Quantum Kernel Learning for Noisy and Adversarial Environment**

针对噪音和对抗环境的稳健分散量子核学习 quant-ph

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13782v1) [paper-pdf](http://arxiv.org/pdf/2504.13782v1)

**Authors**: Wenxuan Ma, Kuan-Cheng Chen, Shang Yu, Mengxiang Liu, Ruilong Deng

**Abstract**: This paper proposes a general decentralized framework for quantum kernel learning (QKL). It has robustness against quantum noise and can also be designed to defend adversarial information attacks forming a robust approach named RDQKL. We analyze the impact of noise on QKL and study the robustness of decentralized QKL to the noise. By integrating robust decentralized optimization techniques, our method is able to mitigate the impact of malicious data injections across multiple nodes. Experimental results demonstrate that our approach maintains high accuracy under noisy quantum operations and effectively counter adversarial modifications, offering a promising pathway towards the future practical, scalable and secure quantum machine learning (QML).

摘要: 本文提出了一个通用的量子核学习（QKL）的分散框架。它对量子噪声具有鲁棒性，也可以设计用于防御对抗性信息攻击，形成一种名为RDQKL的鲁棒方法。我们分析了噪声对QKL的影响，并研究了分散QKL对噪声的鲁棒性。通过集成强大的分散优化技术，我们的方法能够减轻跨多个节点的恶意数据注入的影响。实验结果表明，我们的方法在有噪音的量子操作下保持了高准确性，并有效地对抗对抗修改，为未来实用、可扩展和安全的量子机器学习（QML）提供了一条有希望的途径。



## **2. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2412.14113v2) [paper-pdf](http://arxiv.org/pdf/2412.14113v2)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated with respect to 100 randomly selected target queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是多维载体空间中的一种现象，其中自然分布的单个点与许多其他点异常接近。这是信息检索中一个众所周知的问题，会导致某些项意外（且错误地）看起来与许多查询相关。   在本文中，我们研究攻击者如何利用中心将多模式检索系统中的任何图像或音频输入变成对抗中心。对抗中心可用于注入通用对抗内容（例如，垃圾邮件）将根据数千个不同的查询进行检索，并对与攻击者选择的特定概念相关的查询进行有针对性的攻击。   我们提出了一种创建对抗中心的方法，并在基准多模式检索数据集和由流行的载体数据库Pinecone实现的图像到图像检索系统上评估所得中心。例如，在文本标题到图像检索中，针对100个随机选择的目标查询生成的单个对抗中心被检索为25，000个测试查询中超过21，000个的前1最相关图像（相比之下，最常见的自然中心是仅对102个查询的前1响应），这表明了对抗性中心的强大概括能力。我们还调查了减轻自然中心的技术是否是对抗性中心的有效防御，并表明它们对针对与特定概念相关的查询的中心无效。



## **3. Energy-Latency Attacks via Sponge Poisoning**

通过海绵中毒进行能量潜伏攻击 cs.CR

Paper accepted at Information Sciences journal; 20 pages Keywords:  energy-latency attacks, sponge attack, machine learning security, adversarial  machine learning

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2203.08147v5) [paper-pdf](http://arxiv.org/pdf/2203.08147v5)

**Authors**: Antonio Emanuele Cinà, Ambra Demontis, Battista Biggio, Fabio Roli, Marcello Pelillo

**Abstract**: Sponge examples are test-time inputs optimized to increase energy consumption and prediction latency of deep networks deployed on hardware accelerators. By increasing the fraction of neurons activated during classification, these attacks reduce sparsity in network activation patterns, worsening the performance of hardware accelerators. In this work, we present a novel training-time attack, named sponge poisoning, which aims to worsen energy consumption and prediction latency of neural networks on any test input without affecting classification accuracy. To stage this attack, we assume that the attacker can control only a few model updates during training -- a likely scenario, e.g., when model training is outsourced to an untrusted third party or distributed via federated learning. Our extensive experiments on image classification tasks show that sponge poisoning is effective, and that fine-tuning poisoned models to repair them poses prohibitive costs for most users, highlighting that tackling sponge poisoning remains an open issue.

摘要: Sponge示例是测试时输入，经过优化，以增加部署在硬件加速器上的深度网络的能耗和预测延迟。通过增加分类期间激活的神经元的比例，这些攻击降低了网络激活模式的稀疏性，从而恶化了硬件加速器的性能。在这项工作中，我们提出了一种新的训练时间攻击，称为海绵中毒，其目的是在不影响分类精度的情况下，恶化神经网络在任何测试输入上的能耗和预测延迟。为了进行这种攻击，我们假设攻击者在训练过程中只能控制一些模型更新--这是一种可能的情况，例如，当模型训练外包给不受信任的第三方或通过联邦学习分发时。我们对图像分类任务的广泛实验表明，海绵中毒是有效的，并且微调中毒模型以修复它们对大多数用户来说会带来高昂的成本，这凸显了解决海绵中毒仍然是一个悬而未决的问题。



## **4. Fairness and Robustness in Machine Unlearning**

机器去学习中的公平性和鲁棒性 cs.LG

5 pages

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13610v1) [paper-pdf](http://arxiv.org/pdf/2504.13610v1)

**Authors**: Khoa Tran, Simon S. Woo

**Abstract**: Machine unlearning poses the challenge of ``how to eliminate the influence of specific data from a pretrained model'' in regard to privacy concerns. While prior research on approximated unlearning has demonstrated accuracy and efficiency in time complexity, we claim that it falls short of achieving exact unlearning, and we are the first to focus on fairness and robustness in machine unlearning algorithms. Our study presents fairness Conjectures for a well-trained model, based on the variance-bias trade-off characteristic, and considers their relevance to robustness. Our Conjectures are supported by experiments conducted on the two most widely used model architectures, ResNet and ViT, demonstrating the correlation between fairness and robustness: \textit{the higher fairness-gap is, the more the model is sensitive and vulnerable}. In addition, our experiments demonstrate the vulnerability of current state-of-the-art approximated unlearning algorithms to adversarial attacks, where their unlearned models suffer a significant drop in accuracy compared to the exact-unlearned models. We claim that our fairness-gap measurement and robustness metric should be used to evaluate the unlearning algorithm. Furthermore, we demonstrate that unlearning in the intermediate and last layers is sufficient and cost-effective for time and memory complexity.

摘要: 在隐私问题方面，机器取消学习带来了“如何消除预训练模型中特定数据的影响”的挑战。虽然之前关于近似去学习的研究已经证明了时间复杂性的准确性和效率，但我们声称它未能实现精确去学习，而且我们是第一个关注机器去学习算法中的公平性和鲁棒性的人。我们的研究基于方差偏差权衡特征，提出了训练有素的模型的公平性猜想，并考虑了它们与稳健性的相关性。我们的猜想得到了在两种最广泛使用的模型架构ResNet和ViT上进行的实验的支持，证明了公平性和鲁棒性之间的相关性：\textit{公平差距越大，模型就越敏感和脆弱}。此外，我们的实验还证明了当前最先进的逼近非学习算法对对抗性攻击的脆弱性，与精确非学习模型相比，它们的非学习模型的准确性显着下降。我们声称应该使用我们的公平差距测量和稳健性指标来评估取消学习算法。此外，我们证明，对于时间和内存复杂性来说，在中间层和最后层中取消学习是足够的且具有成本效益的。



## **5. Q-FAKER: Query-free Hard Black-box Attack via Controlled Generation**

Q-FAKER：通过受控生成进行无查询硬黑匣子攻击 cs.CR

NAACL 2025 Findings

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13551v1) [paper-pdf](http://arxiv.org/pdf/2504.13551v1)

**Authors**: CheolWon Na, YunSeok Choi, Jee-Hyong Lee

**Abstract**: Many adversarial attack approaches are proposed to verify the vulnerability of language models. However, they require numerous queries and the information on the target model. Even black-box attack methods also require the target model's output information. They are not applicable in real-world scenarios, as in hard black-box settings where the target model is closed and inaccessible. Even the recently proposed hard black-box attacks still require many queries and demand extremely high costs for training adversarial generators. To address these challenges, we propose Q-faker (Query-free Hard Black-box Attacker), a novel and efficient method that generates adversarial examples without accessing the target model. To avoid accessing the target model, we use a surrogate model instead. The surrogate model generates adversarial sentences for a target-agnostic attack. During this process, we leverage controlled generation techniques. We evaluate our proposed method on eight datasets. Experimental results demonstrate our method's effectiveness including high transferability and the high quality of the generated adversarial examples, and prove its practical in hard black-box settings.

摘要: 人们提出了许多对抗攻击方法来验证语言模型的脆弱性。然而，它们需要大量查询和有关目标模型的信息。即使是黑匣子攻击方法也需要目标模型的输出信息。它们不适用于现实世界场景，例如在目标模型关闭且无法访问的硬黑匣子设置中。即使是最近提出的硬黑匣子攻击仍然需要许多查询，并且需要极高的训练对抗生成器的成本。为了应对这些挑战，我们提出了Q-faker（无查询硬黑匣子攻击者），这是一种新颖且高效的方法，可以在不访问目标模型的情况下生成对抗性示例。为了避免访问目标模型，我们使用代理模型。代理模型为目标不可知攻击生成对抗性句子。在此过程中，我们利用受控发电技术。我们在八个数据集上评估了我们提出的方法。实验结果表明，该方法的有效性，包括高的可移植性和高质量的生成的对抗性的例子，并证明其实用性在硬黑盒设置。



## **6. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

针对顺序推荐系统的少样本模型抽取攻击 cs.LG

It requires substantial modifications.The symbols in the mathematical  formulas are not explained in detail

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2411.11677v2) [paper-pdf](http://arxiv.org/pdf/2411.11677v2)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.

摘要: 在针对顺序推荐系统的对抗性攻击中，模型提取攻击代表了一种在没有先验知识的情况下攻击顺序推荐模型的方法。现有的研究主要集中在对手通过无数据模型提取执行黑匣子攻击。然而，关于对手能够获得少量原始数据（10%甚至更少）开发代理模型的文献中仍然存在显着差距。也就是说，如何在少镜头数据场景的背景下构建具有高功能相似性的代理模型的挑战仍然是一个需要解决的问题。本研究通过引入一种针对顺序排序器的新型少镜头模型提取框架来解决这一差距，该框架旨在利用少镜头数据构建更好的代理模型。提出的少镜头模型提取框架由两个部分组成：自回归增强生成策略和双向修复损失促进模型蒸馏过程。具体来说，为了生成非常接近原始数据分布的合成数据，自回归增强生成策略集成了用于提取固有依赖关系的概率交互采样器和用于描述用户行为模式的合成决定因素信号模块。随后，针对推荐列表之间的差异，将双向修复损失设计为辅助损失，以纠正代理模型的错误预测，有效地将知识从受害者模型转移到代理模型。对三个数据集的实验表明，提出的少镜头模型提取框架可以产生更好的代理模型。



## **7. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.

摘要: 假新闻的传播伤害了个人，并提出了必须解决的严重社会挑战。尽管已经开发了许多算法和有洞察力的功能来检测假新闻，但其中许多功能都可以通过风格转换攻击来操纵，特别是随着高级语言模型的出现，使其更难与真实新闻区分开来。这项研究提出了对抗性风格增强AdStyle，旨在训练假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。主要机制涉及战略性地使用LLM来自动生成多样化且连贯的风格转换攻击提示阵列，从而增强检测器特别具有挑战性的提示的生成。实验表明，当对假新闻基准数据集进行评估时，我们的增强策略显着提高了鲁棒性和检测性能。



## **8. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **9. EXAM: Exploiting Exclusive System-Level Cache in Apple M-Series SoCs for Enhanced Cache Occupancy Attacks**

EXAM：利用Apple M系列SOC中的独占系统级缓存进行增强型缓存占用攻击 cs.CR

Accepted to ACM ASIA CCS 2025

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13385v1) [paper-pdf](http://arxiv.org/pdf/2504.13385v1)

**Authors**: Tianhong Xu, Aidong Adam Ding, Yunsi Fei

**Abstract**: Cache occupancy attacks exploit the shared nature of cache hierarchies to infer a victim's activities by monitoring overall cache usage, unlike access-driven cache attacks that focus on specific cache lines or sets. There exists some prior work that target the last-level cache (LLC) of Intel processors, which is inclusive of higher-level caches, and L2 caches of ARM systems. In this paper, we target the System-Level Cache (SLC) of Apple M-series SoCs, which is exclusive to higher-level CPU caches. We address the challenges of the exclusiveness and propose a suite of SLC-cache occupancy attacks, the first of its kind, where an adversary can monitor GPU and other CPU cluster activities from their own CPU cluster. We first discover the structure of SLC in Apple M1 SOC and various policies pertaining to access and sharing through reverse engineering. We propose two attacks against websites. One is a coarse-grained fingerprinting attack, recognizing which website is accessed based on their different GPU memory access patterns monitored through the SLC occupancy channel. The other attack is a fine-grained pixel stealing attack, which precisely monitors the GPU memory usage for rendering different pixels, through the SLC occupancy channel. Third, we introduce a novel screen capturing attack which works beyond webpages, with the monitoring granularity of 57 rows of pixels (there are 1600 rows for the screen). This significantly expands the attack surface, allowing the adversary to retrieve any screen display, posing a substantial new threat to system security. Our findings reveal critical vulnerabilities in Apple's M-series SoCs and emphasize the urgent need for effective countermeasures against cache occupancy attacks in heterogeneous computing environments.

摘要: 缓存占用率攻击利用缓存层次结构的共享性质，通过监控总体缓存使用情况来推断受害者的活动，这与专注于特定缓存行或集的访问驱动缓存攻击不同。以前有一些针对英特尔处理器的最后一级缓存（LLC）的工作，其中包括更高级的缓存和ARM系统的L2缓存。在本文中，我们的目标是Apple M系列SOC的系统级缓存（SLC），该缓存专为更高级的中央处理器缓存。我们解决了独占性的挑战，并提出了一套SLC缓存占用率攻击，这是此类攻击中的第一个，对手可以通过其自己的中央处理器集群监控图形处理器和其他中央处理器集群活动。我们首先发现Apple M1 SOC中SLC的结构以及与通过反向工程访问和共享相关的各种政策。我们提出了两种针对网站的攻击。一种是粗粒度指纹攻击，根据通过SLC占用通道监控的不同图形处理器内存访问模式来识别访问哪个网站。另一种攻击是细粒度像素窃取攻击，该攻击通过SLC占用通道精确监控渲染不同像素的图形处理器内存使用情况。第三，我们引入了一种新颖的屏幕捕获攻击，其适用于网页之外，监控粒度为57行像素（屏幕有1600行）。这显着扩大了攻击面，允许对手检索任何屏幕显示，对系统安全构成了重大的新威胁。我们的调查结果揭示了苹果M系列SOC中的关键漏洞，并强调迫切需要针对异类计算环境中的缓存占用攻击采取有效的应对措施。



## **10. DYNAMITE: Dynamic Defense Selection for Enhancing Machine Learning-based Intrusion Detection Against Adversarial Attacks**

CLARITE：动态防御选择，以增强基于机器学习的入侵检测对抗性攻击 cs.CR

Accepted by the IEEE/ACM Workshop on the Internet of Safe Things  (SafeThings 2025)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13301v1) [paper-pdf](http://arxiv.org/pdf/2504.13301v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Elvin Li, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) has introduced substantial security vulnerabilities, highlighting the need for robust Intrusion Detection Systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we propose Dynamite, a dynamic defense selection framework that enhances ML-IDS by intelligently identifying and deploying the most suitable defense using a machine learning-driven selection mechanism. Our results demonstrate that Dynamite achieves a 96.2% reduction in computational time compared to the Oracle, significantly decreasing computational overhead while preserving strong prediction performance. Dynamite also demonstrates an average F1-score improvement of 76.7% over random defense and 65.8% over the best static state-of-the-art defense.

摘要: 物联网（IOT）的迅速普及引入了大量的安全漏洞，凸显了对强大的入侵检测系统（IDS）的需求。基于机器学习的入侵检测系统（ML-IDS）显着提高了威胁检测能力;然而，它们仍然极易受到对抗性攻击。虽然已经提出了多种防御机制来增强ML-IDS的弹性，但仍然缺乏一种系统性的方法来选择针对特定对抗攻击的最有效防御。为了应对这一挑战，我们提出了Dynamite，这是一个动态防御选择框架，它通过使用机器学习驱动的选择机制智能识别和部署最合适的防御来增强ML-IDS。我们的结果表明，与Oracle相比，Dynamite的计算时间减少了96.2%，显着降低了计算负担，同时保持了强劲的预测性能。Dynamite还表明，F1平均得分比随机防守提高了76.7%，比最佳静态最先进防御提高了65.8%。



## **11. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

通过简单的自适应攻击越狱领先的安全一致LLM cs.CR

Accepted at ICLR 2025. Updates in the v3: GPT-4o and Claude 3.5  Sonnet results, improved writing. Updates in the v2: more models (Llama3,  Phi-3, Nemotron-4-340B), jailbreak artifacts for all attacks are available,  evaluation with different judges (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots, ablation on the suffix length for random  search), examples of jailbroken generation

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2404.02151v4) [paper-pdf](http://arxiv.org/pdf/2404.02151v4)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全一致的LLM也无法抵御简单的自适应越狱攻击。首先，我们演示了如何成功利用对logprob的访问进行越狱：我们最初设计一个对抗提示模板（有时适合目标LLM），然后对后缀应用随机搜索以最大化目标logprob（例如，令牌“Sure”），可能会多次重新启动。通过这种方式，我们在Vicuna-13 B、Mistral-7 B、Phi-3-Mini、Nemotron-4- 340 B、Llama-2-Chat-7 B/13 B/70 B、Llama-3-Direct-8B、Gemma-7 B、GPT-3.5、GPT-4 o和来自HarmBench的R2 D2上实现了100%的攻击成功率，该公司接受了针对GCG攻击的对抗训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有Claude模型（不会暴露logprobs）。此外，我们还展示了如何对一组受限制的令牌使用随机搜索来在中毒模型中寻找特洛伊木马字符串--这项任务与越狱有许多相似之处--该算法使我们在SaTML ' 24特洛伊木马检测竞赛中获得第一名。这些攻击背后的共同主题是自适应性至关重要：不同的模型容易受到不同提示模板的影响（例如，R2 D2对上下文学习提示非常敏感），一些模型根据其API存在独特的漏洞（例如，为Claude预填充），在某些设置中，基于先验知识限制令牌搜索空间至关重要（例如，用于木马检测）。出于重现性的目的，我们在https://github.com/tml-epfl/llm-adaptive-attacks上提供了JailbreakBench格式的代码、日志和越狱工件。



## **12. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

Chypnosis：使用基于欠压的静态侧通道攻击进行秘密提取 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.11633v2) [paper-pdf](http://arxiv.org/pdf/2504.11633v2)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.

摘要: 越来越多的静态物理侧通道攻击允许对手通过探测电路的持续状态来提取秘密。激光逻辑状态成像（LLSI）、阻抗分析（IA）和静态功率分析等技术都属于这一类。这些攻击要求目标数据在特定的持续时间内保持不变，这通常需要停止电路的时钟。有些方法还依赖于调制芯片的电源电压来探测电路。然而，对时钟或电压的篡改通常被认为是可检测的，因为安全芯片通常部署传感器，在检测到此类异常时擦除敏感数据。此外，许多安全设备使用内部时钟源，使得外部时钟控制不可行。在这项工作中，我们引入了一类新型的静态侧通道攻击，称为Chypnosis，它使对手能够通过快速欠电压诱导休眠状态来冻结芯片的内部时钟，然后使用静态侧通道提取秘密。我们证明，通过将芯片的电压快速降低到标准名义水平以下，攻击者可以绕过时钟和电压传感器，并将芯片置于所谓的停电条件，其中芯片的晶体管停止切换，但易失性存储器（例如，人字拖和RAM）仍然保留其数据。我们通过将AMD VGA置于休眠状态来测试对它们的攻击。我们表明，不仅所有时钟源都被停用，而且各种时钟和电压传感器也无法检测到篡改事件。随后，我们展示了使用两种静态攻击（LLSI和IA）从休眠芯片中成功恢复秘密位的方法。最后，我们讨论了可以整合到未来设计中的潜在对策。



## **13. Strategic Planning of Stealthy Backdoor Attacks in Markov Decision Processes**

马尔科夫决策过程中隐形后门攻击的战略规划 eess.SY

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13276v1) [paper-pdf](http://arxiv.org/pdf/2504.13276v1)

**Authors**: Xinyi Wei, Shuo Han, Ahmed H. Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: This paper investigates backdoor attack planning in stochastic control systems modeled as Markov Decision Processes (MDPs). In a backdoor attack, the adversary provides a control policy that behaves well in the original MDP to pass the testing phase. However, when such a policy is deployed with a trigger policy, which perturbs the system dynamics at runtime, it optimizes the attacker's objective instead. To solve jointly the control policy and its trigger, we formulate the attack planning problem as a constrained optimal planning problem in an MDP with augmented state space, with the objective to maximize the attacker's total rewards in the system with an activated trigger, subject to the constraint that the control policy is near optimal in the original MDP. We then introduce a gradient-based optimization method to solve the optimal backdoor attack policy as a pair of coordinated control and trigger policies. Experimental results from a case study validate the effectiveness of our approach in achieving stealthy backdoor attacks.

摘要: 研究了随机控制系统中的后门攻击规划问题。在后门攻击中，对手提供了一个在原始MDP中表现良好的控制策略，以通过测试阶段。然而，当这样的策略与在运行时扰乱系统动态的触发策略一起部署时，它反而优化了攻击者的目标。为了共同解决的控制策略和它的触发器，我们制定的攻击规划问题作为一个有约束的最优规划问题的MDP与增广的状态空间，目标是最大限度地提高攻击者的总回报在系统中激活触发器，受约束的控制策略是接近最优的原始MDP。然后，我们引入一种基于梯度的优化方法来解决最优后门攻击策略，作为一对协调的控制和触发策略。案例研究的实验结果验证了我们的方法在实现隐形后门攻击方面的有效性。



## **14. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5  Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.11969v4) [paper-pdf](http://arxiv.org/pdf/2407.11969v4)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝培训被广泛用于防止LLM产生有害、不良或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括差距：简单地用过去时重新表达有害的请求（例如，“如何制作燃烧弹？”到“人们是如何制作燃烧弹的？”）通常足以越狱许多最先进的法学硕士。我们在Llama-3 8B、Claude-3.5十四行诗、GPT-3.5 Turbo、Gemma-2 9 B、Phi-3-Mini、GPT-4 o mini、GPT-4 o mini、o 1-mini、o 1-预览和R2D2模型上系统地评估了该方法，使用GPT-3.5 Turbo作为重新制定模型。例如，这种对GPT-4 o的简单攻击的成功率从使用直接请求的1%增加到使用来自JailbreakBench的有害请求的20次过去时重新表述尝试（以GPT-4作为越狱法官）的88%。有趣的是，我们还发现未来时的重新表述效果不太好，这表明拒绝护栏往往会考虑过去的历史问题而不是假设的未来问题。此外，我们的微调GPT-3.5涡轮实验表明，防御过去的改写是可行的，过去时态的例子显式地包括在微调数据。总的来说，我们的研究结果强调了广泛使用的对齐技术-如SFT，RLHF和对抗训练-用于对齐所研究的模型可能是脆弱的，并且并不总是按预期进行推广。我们在https://github.com/tml-epfl/llm-past-tense上提供代码和越狱工件。



## **15. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

通过自然和对抗性局部腐蚀对DNN的空间鲁棒性进行基准测试 cs.CV

Under review

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.01632v2) [paper-pdf](http://arxiv.org/pdf/2504.01632v2)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were exploited to evaluate 14 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.

摘要: DNN的鲁棒性是安全关键型应用程序中的一个关键因素，特别是在可能出现局部损坏的复杂动态环境中。虽然以前的研究已经评估了语义分割（SS）模型在整个图像自然或对抗性腐败下的鲁棒性，但对密集视觉模型在局部腐败下的空间鲁棒性的全面调查仍然没有得到充分的探索。本文填补了这一空白，引入了专门的度量基准分割模型的空间鲁棒性，以及评估框架，以评估本地化的腐败的影响。此外，我们揭示了固有的复杂性，使用一个单一的本地化对抗扰动的特征最坏情况下的鲁棒性。为了解决这个问题，我们提出了区域感知的多攻击对抗分析，这种方法可以更深入地理解模型针对应用于特定区域的对抗扰动的稳健性。利用提出的指标和分析来评估驾驶场景中的14个细分模型，揭示了对自然形式和对抗形式的局部腐败影响的关键见解。结果表明，模型对这两种类型的威胁的反应不同;例如，基于变换器的分割模型对局部自然破坏表现出显着的鲁棒性，但极易受到对抗性破坏的影响，而基于CNN的模型则反之亦然。因此，我们还通过集成模型解决了平衡对自然和对抗局部破坏的鲁棒性的挑战，从而实现更广泛的威胁覆盖范围并提高密集视觉任务的可靠性。



## **16. AHSG: Adversarial Attack on High-level Semantics in Graph Neural Networks**

AHSG：对图神经网络中高级语义的对抗攻击 cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2412.07468v2) [paper-pdf](http://arxiv.org/pdf/2412.07468v2)

**Authors**: Kai Yuan, Jiahao Zhang, Yidi Wang, Xiaobing Pei

**Abstract**: Adversarial attacks on Graph Neural Networks aim to perturb the performance of the learner by carefully modifying the graph topology and node attributes. Existing methods achieve attack stealthiness by constraining the modification budget and differences in graph properties. However, these methods typically disrupt task-relevant primary semantics directly, which results in low defensibility and detectability of the attack. In this paper, we propose an Adversarial Attack on High-level Semantics for Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. By combining latent representations with shared primary semantics, our model retains detectable attributes and relational patterns of the original graph while leveraging more subtle changes to carry out the attack. Then we use the Projected Gradient Descent algorithm to map the latent representations with attack effects to the adversarial graph. Through experiments on robust graph deep learning models equipped with defense strategies, we demonstrate that AHSG outperforms other state-of-the-art methods in attack effectiveness. Additionally, using Contextual Stochastic Block Models to detect the attacked graph further validates that our method preserves the primary semantics of the graph.

摘要: 对图神经网络的对抗攻击旨在通过仔细修改图布局和节点属性来扰乱学习器的性能。现有方法通过限制修改预算和图属性的差异来实现攻击隐蔽性。然而，这些方法通常直接破坏与任务相关的主要语义，从而导致攻击的防御性和可检测性较低。本文提出了一种对图神经网络高级语义的对抗攻击（AHSG），这是一种确保主要语义保留的图结构攻击模型。通过将潜在表示与共享的主要语义相结合，我们的模型保留了原始图的可检测属性和关系模式，同时利用更微妙的变化来执行攻击。然后我们使用投影梯度下降算法将具有攻击效果的潜在表示映射到对抗图。通过对配备防御策略的稳健图深度学习模型的实验，我们证明AHSG在攻击有效性方面优于其他最先进的方法。此外，使用上下文随机块模型来检测受攻击的图进一步验证了我们的方法保留了图的主要语义。



## **17. Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models**

数据重复对基于深度神经网络的图像分类器的影响：稳健模型与标准模型 cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.00638v2) [paper-pdf](http://arxiv.org/pdf/2504.00638v2)

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.   In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy.

摘要: 机器学习模型针对对抗性攻击的准确性和稳健性受到训练数据质量、模型架构、训练过程和部署环境等因素的显着影响。近年来，训练集中的重复数据，尤其是语言模型中的重复数据，引起了相当大的关注。事实证明，去重可以增强语言模型中的训练性能和模型准确性。虽然训练图像分类器深度神经网络（DNN）中数据质量的重要性已被广泛认识到，但训练集中重复图像对模型概括性和性能的影响却很少受到关注。   本文中，我们解决了这一差距，并对图像分类中重复的影响进行了全面的研究。我们的分析表明，训练集中重复图像的存在不仅会对模型训练的效率产生负面影响，还会导致图像分类器的准确性较低。当重复的数据在类别中不一致时，或者当重复（无论是一致还是不一致）发生在对抗训练模型的训练集中时，重复对准确性的负面影响尤其明显。即使以统一的方式选择重复样本，增加重复数量也不会导致准确性的显着提高。



## **18. A Survey and Evaluation of Adversarial Attacks for Object Detection**

目标检测中的对抗性攻击综述与评价 cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2408.01934v5) [paper-pdf](http://arxiv.org/pdf/2408.01934v5)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.

摘要: 深度学习模型在计算机视觉任务中实现了非凡的准确性，但仍然容易受到对抗性示例的影响--对输入图像精心设计的扰动，可能会欺骗这些模型做出自信但不正确的预测。该漏洞在自动驾驶汽车、安全监控和安全关键检查系统等高风险应用中构成了重大风险。虽然现有文献广泛涵盖了图像分类中的对抗攻击，但对对象检测系统上的此类攻击的全面分析仍然有限。本文提出了一种新颖的分类框架，用于对特定于对象检测架构的对抗性攻击进行分类，综合了现有的鲁棒性指标，并对流行对象检测模型（包括传统检测器和具有视觉语言预训练的现代检测器）的最新攻击方法进行了全面的实证评估。通过严格分析开源攻击实现及其在不同检测架构中的有效性，我们获得了对攻击特征的关键见解。此外，我们描述了关键的研究差距和新出现的挑战，以指导未来的调查，以确保对象检测系统免受对抗性威胁。我们的研究结果为开发更强大的检测模型奠定了基础，同时强调了在这个快速发展的领域迫切需要标准化的评估协议。



## **19. Privacy Protection Against Personalized Text-to-Image Synthesis via Cross-image Consistency Constraints**

通过跨图像一致性约束针对个性化文本到图像合成的隐私保护 cs.CV

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12747v1) [paper-pdf](http://arxiv.org/pdf/2504.12747v1)

**Authors**: Guanyu Wang, Kailong Wang, Yihao Huang, Mingyi Zhou, Zhang Qing cnwatcher, Geguang Pu, Li Li

**Abstract**: The rapid advancement of diffusion models and personalization techniques has made it possible to recreate individual portraits from just a few publicly available images. While such capabilities empower various creative applications, they also introduce serious privacy concerns, as adversaries can exploit them to generate highly realistic impersonations. To counter these threats, anti-personalization methods have been proposed, which add adversarial perturbations to published images to disrupt the training of personalization models. However, existing approaches largely overlook the intrinsic multi-image nature of personalization and instead adopt a naive strategy of applying perturbations independently, as commonly done in single-image settings. This neglects the opportunity to leverage inter-image relationships for stronger privacy protection. Therefore, we advocate for a group-level perspective on privacy protection against personalization. Specifically, we introduce Cross-image Anti-Personalization (CAP), a novel framework that enhances resistance to personalization by enforcing style consistency across perturbed images. Furthermore, we develop a dynamic ratio adjustment strategy that adaptively balances the impact of the consistency loss throughout the attack iterations. Extensive experiments on the classical CelebHQ and VGGFace2 benchmarks show that CAP substantially improves existing methods.

摘要: 扩散模型和个性化技术的快速发展使得仅从少数公开图像中重建个人肖像成为可能。虽然这些功能支持各种创意应用程序，但它们也带来了严重的隐私问题，因为对手可以利用它们来生成高度真实的模仿。为了应对这些威胁，人们提出了反个性化方法，这些方法向已发布的图像添加对抗性扰动，以扰乱个性化模型的训练。然而，现有的方法在很大程度上忽视了个性化固有的多图像本质，而是采用独立应用扰动的天真策略，就像在单图像设置中常见的那样。这忽视了利用图像间关系来加强隐私保护的机会。因此，我们主张以群体层面的视角来保护隐私，防止个性化。具体来说，我们引入了跨图像反个性化（CAP），这是一种新颖的框架，通过在受干扰的图像中强制执行风格一致性来增强对个性化的抵抗力。此外，我们开发了一种动态比率调整策略，该策略自适应地平衡整个攻击迭代期间一致性损失的影响。对经典CelebHQ和VGFace2基准测试的广泛实验表明，CAP极大地改进了现有方法。



## **20. Adversary-Augmented Simulation for Fairness Evaluation and Defense in Hyperledger Fabric**

Hyperledger结构中公平性评估和防御的对手增强模拟 cs.CR

20 pages, 14 figures. arXiv admin note: text overlap with  arXiv:2403.14342

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12733v1) [paper-pdf](http://arxiv.org/pdf/2504.12733v1)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Pierre-Yves Piriou, Sara Tucci-Piergiovanni

**Abstract**: This paper presents an adversary model and a simulation framework specifically tailored for analyzing attacks on distributed systems composed of multiple distributed protocols, with a focus on assessing the security of blockchain networks. Our model classifies and constrains adversarial actions based on the assumptions of the target protocols, defined by failure models, communication models, and the fault tolerance thresholds of Byzantine Fault Tolerant (BFT) protocols. The goal is to study not only the intended effects of adversarial strategies but also their unintended side effects on critical system properties. We apply this framework to analyze fairness properties in a Hyperledger Fabric (HF) blockchain network. Our focus is on novel fairness attacks that involve coordinated adversarial actions across various HF services. Simulations show that even a constrained adversary can violate fairness with respect to specific clients (client fairness) and impact related guarantees (order fairness), which relate the reception order of transactions to their final order in the blockchain. This paper significantly extends our previous work by introducing and evaluating a mitigation mechanism specifically designed to counter transaction reordering attacks. We implement and integrate this defense into our simulation environment, demonstrating its effectiveness under diverse conditions.

摘要: 本文提出了一个对手模型和一个专门为分析对由多个分布式协议组成的分布式系统的攻击而定制的模拟框架，重点是评估区块链网络的安全性。我们的模型根据目标协议的假设对对抗行为进行分类和约束，目标协议由故障模型、通信模型和拜占庭式故障容忍（BFT）协议的故障容忍阈值定义。目标不仅是研究对抗策略的预期影响，还研究其对关键系统属性的意外副作用。我们应用这个框架来分析Hyperledger Fabric（HF）区块链网络中的公平属性。我们的重点是新颖的公平攻击，涉及各种HF服务之间的协调对抗行动。模拟表明，即使是受约束的对手也可能违反对特定客户的公平性（客户公平性）并影响相关保证（订单公平性），这些保证将交易的接收顺序与区块链中的最终顺序联系起来。本文通过引入和评估专门设计用于对抗事务重新排序攻击的缓解机制，显着扩展了我们之前的工作。我们实施这种防御并将其集成到我们的模拟环境中，展示其在不同条件下的有效性。



## **21. Quantum Computing Supported Adversarial Attack-Resilient Autonomous Vehicle Perception Module for Traffic Sign Classification**

量子计算支持的对抗性攻击-弹性自主车辆感知模块用于交通标志分类 cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12644v1) [paper-pdf](http://arxiv.org/pdf/2504.12644v1)

**Authors**: Reek Majumder, Mashrur Chowdhury, Sakib Mahmud Khan, Zadid Khan, Fahim Ahmad, Frank Ngeni, Gurcan Comert, Judith Mwakalonge, Dimitra Michalaka

**Abstract**: Deep learning (DL)-based image classification models are essential for autonomous vehicle (AV) perception modules since incorrect categorization might have severe repercussions. Adversarial attacks are widely studied cyberattacks that can lead DL models to predict inaccurate output, such as incorrectly classified traffic signs by the perception module of an autonomous vehicle. In this study, we create and compare hybrid classical-quantum deep learning (HCQ-DL) models with classical deep learning (C-DL) models to demonstrate robustness against adversarial attacks for perception modules. Before feeding them into the quantum system, we used transfer learning models, alexnet and vgg-16, as feature extractors. We tested over 1000 quantum circuits in our HCQ-DL models for projected gradient descent (PGD), fast gradient sign attack (FGSA), and gradient attack (GA), which are three well-known untargeted adversarial approaches. We evaluated the performance of all models during adversarial attacks and no-attack scenarios. Our HCQ-DL models maintain accuracy above 95\% during a no-attack scenario and above 91\% for GA and FGSA attacks, which is higher than C-DL models. During the PGD attack, our alexnet-based HCQ-DL model maintained an accuracy of 85\% compared to C-DL models that achieved accuracies below 21\%. Our results highlight that the HCQ-DL models provide improved accuracy for traffic sign classification under adversarial settings compared to their classical counterparts.

摘要: 基于深度学习（DL）的图像分类模型对于自动驾驶汽车（AV）感知模块至关重要，因为不正确的分类可能会产生严重的影响。对抗性攻击是一种被广泛研究的网络攻击，它可能导致DL模型预测不准确的输出，例如自动驾驶汽车的感知模块错误地分类交通标志。在这项研究中，我们创建并比较了混合经典-量子深度学习（HCQ-DL）模型与经典深度学习（C-DL）模型，以证明感知模块对抗对抗攻击的鲁棒性。在将它们输入量子系统之前，我们使用了迁移学习模型alexnet和vgg-16作为特征提取器。我们在HCQ-DL模型中测试了1000多个量子电路，用于投影梯度下降（PVD）、快速梯度符号攻击（FGSA）和梯度攻击（GA），这是三种众所周知的无目标对抗方法。我们评估了所有模型在对抗攻击和无攻击场景下的性能。我们的HCQ-DL模型在无攻击场景下保持在95%以上的准确性，在GA和FGSA攻击时保持在91%以上，高于C-DL模型。在PVD攻击期间，我们基于alexnet的HCQ-DL模型保持了85%的准确性，而C-DL模型的准确性低于21%。我们的结果强调，与经典模型相比，HCQ-DL模型在对抗环境下为交通标志分类提供了更高的准确性。



## **22. ControlNET: A Firewall for RAG-based LLM System**

Control NET：基于RAG的LLM系统的防火墙 cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.

摘要: 检索增强生成（RAG）显着增强了大型语言模型（LLM）的事实准确性和领域适应性。这一进步使它们能够在医疗保健、金融和企业应用程序等敏感领域广泛部署。RAG通过整合外部知识来缓解幻觉，但也会带来隐私风险和安全风险，尤其是数据泄露风险和数据中毒风险。虽然最近的研究探索了即时注射和中毒攻击，但在控制入站和出站查询流以减轻这些威胁的全面研究方面仍然存在显着差距。在本文中，我们提出了一种人工智能防火墙Controller NET，旨在保护基于RAG的LLM系统免受这些漏洞的影响。ControlNET通过利用激活转变现象来检测对抗性查询并通过语义分歧减轻其影响来控制查询流。我们使用最先进的开源LLM（Llama 3、Vicuna和Mistral）对四个不同的基准数据集（包括Mmarco、HotpotQA、FinQA和MedalSys）进行全面实验。我们的结果表明，ControlNET在检测和缓解安全威胁同时保持系统无害性方面达到了超过0.909 AUROC。总的来说，ControlNET提供了一种有效、健壮、无害的防御机制，标志着基于RAG的LLM系统安全部署的重大进步。



## **23. On the completeness of several fortification-interdiction games in the Polynomial Hierarchy**

关于多项等级中几个防御-拦截游戏的完整性 cs.CC

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.01756v2) [paper-pdf](http://arxiv.org/pdf/2406.01756v2)

**Authors**: Alberto Boggio Tomasaz, Margarida Carvalho, Roberto Cordone, Pierre Hosteins

**Abstract**: Fortification-interdiction games are tri-level adversarial games where two opponents act in succession to protect, disrupt and simply use an infrastructure for a specific purpose. Many such games have been formulated and tackled in the literature through specific algorithmic methods, however very few investigations exist on the completeness of such fortification problems in order to locate them rigorously in the polynomial hierarchy. We clarify the completeness status of several well-known fortification problems, such as the Tri-level Interdiction Knapsack Problem with unit fortification and attack weights, the Max-flow Interdiction Problem and Shortest Path Interdiction Problem with Fortification, the Multi-level Critical Node Problem with unit weights, as well as a well-studied electric grid defence planning problem. For all of these problems, we prove their completeness either for the $\Sigma^p_2$ or the $\Sigma^p_3$ class of the polynomial hierarchy. We also prove that the Multi-level Fortification-Interdiction Knapsack Problem with an arbitrary number of protection and interdiction rounds and unit fortification and attack weights is complete for any level of the polynomial hierarchy, therefore providing a useful basis for further attempts at proving the completeness of protection-interdiction games at any level of said hierarchy.

摘要: 防御拦截游戏是三级对抗游戏，两个对手连续采取行动保护、破坏并简单地将基础设施用于特定目的。文献中已经通过特定的算法方法制定和解决了许多此类游戏，但很少有关于此类防御问题的完整性的研究，以将它们严格地定位在多元分层结构中。我们澄清了几个著名的防御问题的完整性状态，例如具有单位防御和攻击权重的三层拦截背包问题、具有防御的最大流拦截问题和最短路径拦截问题、具有单位权重的多层关键节点问题，以及一个经过充分研究的电网防御规划问题。对于所有这些问题，我们证明了它们对于$\Sigma ' p_2 $或$\Sigma ' p_3 $类的完整性。我们还证明，具有任意数量的保护和拦截回合以及单位防御和攻击权重的多层防御-拦截背包问题对于多项多元分层结构的任何级别来说都是完整的，因此为进一步尝试证明所述分层结构的任何级别上的保护-拦截游戏的完整性提供了有用的基础。



## **24. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

SAIF：稀疏对抗和不可感知攻击框架 cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2212.07495v3) [paper-pdf](http://arxiv.org/pdf/2212.07495v3)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Masih Eskander, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，向图像添加计算出的微小失真可能会欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新型攻击技术，称为稀疏对抗和可解释攻击框架（SAIF）。具体来说，我们设计了不可感知的攻击，其中包含少量像素的低幅度扰动，并利用这些稀疏攻击来揭示分类器的漏洞。我们使用Frank-Wolfe（条件梯度）算法同时优化有界幅度和稀疏性的攻击扰动，并具有$O（1/\SQRT{T}）$收敛。经验结果表明，SAIF可以计算高度不可感知且可解释的对抗示例，并且在ImageNet数据集上优于最先进的稀疏攻击方法。



## **25. Human Aligned Compression for Robust Models**

鲁棒模型的人类一致压缩 cs.CV

Presented at the Workshop AdvML at CVPR 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12255v1) [paper-pdf](http://arxiv.org/pdf/2504.12255v1)

**Authors**: Samuel Räber, Andreas Plesner, Till Aczel, Roger Wattenhofer

**Abstract**: Adversarial attacks on image models threaten system robustness by introducing imperceptible perturbations that cause incorrect predictions. We investigate human-aligned learned lossy compression as a defense mechanism, comparing two learned models (HiFiC and ELIC) against traditional JPEG across various quality levels. Our experiments on ImageNet subsets demonstrate that learned compression methods outperform JPEG, particularly for Vision Transformer architectures, by preserving semantically meaningful content while removing adversarial noise. Even in white-box settings where attackers can access the defense, these methods maintain substantial effectiveness. We also show that sequential compression--applying rounds of compression/decompression--significantly enhances defense efficacy while maintaining classification performance. Our findings reveal that human-aligned compression provides an effective, computationally efficient defense that protects the image features most relevant to human and machine understanding. It offers a practical approach to improving model robustness against adversarial threats.

摘要: 对图像模型的对抗性攻击通过引入导致错误预测的不可感知的扰动来威胁系统的鲁棒性。我们研究了作为防御机制的人类对齐习得有损压缩，将两种习得模型（HiFiC和ELIC）与不同质量级别的传统JPEG进行比较。我们对ImageNet子集的实验表明，学习的压缩方法通过保留语义有意义的内容同时去除对抗性噪音而优于JPEG，特别是对于Vision Transformer架构。即使在攻击者可以访问防御的白盒环境中，这些方法也能保持相当大的有效性。我们还表明，顺序压缩-应用轮的压缩/解压缩-显着提高防御效率，同时保持分类性能。我们的研究结果表明，人类对齐的压缩提供了一种有效的，计算效率高的防御，可以保护与人类和机器理解最相关的图像特征。它提供了一种实用的方法来提高模型对对抗性威胁的鲁棒性。



## **26. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

LLM护栏中的快速注射和越狱检测 cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六种主要保护系统（包括微软的Azure Promise Shield和Meta的Promise Guard）进行测试，我们表明这两种方法都可以用来逃避检测，同时保持对抗性效用，在某些情况下实现高达100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **27. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全一致和取消学习 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 当前对LLM对抗鲁棒性的研究重点是自然语言空间中的离散输入操纵，其可以直接转移到闭源模型。然而，这种方法忽视了开源模型的稳定发展。随着开源模型功能的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM量身定制的攻击在很大程度上仍然未被探索。我们解决了这一研究空白并提出了嵌入空间攻击，该攻击直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地规避模型对齐并触发有害行为。此外，我们在取消学习的背景下提出了一种新颖的威胁模型，并表明嵌入空间攻击可以从多个数据集和模型中未学习的LLM中提取据称已删除的信息。我们的研究结果强调将空间攻击嵌入到开源LLM中作为重要威胁模型。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **28. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

具有不确定节点特征和不确定图结构的图卷积网络的形式化验证 cs.LG

published at Transactions on Machine Learning Research (TMLR) 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2404.15065v2) [paper-pdf](http://arxiv.org/pdf/2404.15065v2)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.

摘要: 图神经网络因其处理以图结构化的数据的独特能力而在机器学习领域变得越来越受欢迎。它们还应用于固有地发生扰动的安全关键环境中。然而，这些扰动需要我们在将神经网络部署到安全关键环境中之前对其进行正式验证，因为神经网络容易受到对抗性攻击。虽然存在关于神经网络形式验证的研究，但还没有任何工作验证通用图卷积网络架构的稳健性，因为节点特征和多个消息传递步骤中的图结构存在不确定性。这项工作通过使用（矩阵）多项分区的可达性分析明确保留基础计算中所有元素的非凸依赖性来解决这一研究空白。我们在三个流行的基准数据集上展示了我们的方法。



## **29. Attribute Inference Attacks for Federated Regression Tasks**

针对联邦回归任务的属性推理攻击 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2411.12697v2) [paper-pdf](http://arxiv.org/pdf/2411.12697v2)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.

摘要: 联合学习（FL）使多个客户端（如手机和物联网设备）能够协作训练全局机器学习模型，同时保持数据本地化。然而，最近的研究表明，FL的训练阶段容易受到重建攻击，如属性推理攻击（AIA），其中对手利用交换的消息和辅助公共信息来发现目标客户端的敏感属性。虽然这些攻击已经在分类任务的背景下进行了广泛的研究，但它们对回归任务的影响在很大程度上仍未被探索。在本文中，我们通过提出专门为FL环境中的回归任务设计的新的基于模型的AIA来解决这一差距。我们的方法考虑了对手可以窃听交换的消息或直接干扰训练过程的场景。我们使用现实世界的数据集针对最先进的方法对我们提出的攻击进行基准测试。结果表明重建准确性显着提高，特别是在FL中常见的异类客户端数据集中。我们基于模型的AIAs的功效使它们成为经验量化联邦回归任务隐私泄露的更好候选者。



## **30. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

RL SA-PFL：隐私保护联邦学习中具有模型不一致性检测的鲁棒轻量级安全聚合 cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2502.08989v2) [paper-pdf](http://arxiv.org/pdf/2502.08989v2)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.

摘要: 联合学习（FL）允许用户通过仅共享本地模型来协作训练全球机器学习模型，而无需将其私人数据暴露给中央服务器。这种分布式学习在数据隐私至关重要的场景中特别有吸引力，并且引起了行业和学术界的高度关注。然而，研究揭示了FL中的隐私漏洞，对手可能会从共享模型参数中推断敏感信息。在本文中，我们提出了一种高效的基于掩蔽的安全聚合方案，利用轻量级加密基元来降低隐私风险。与现有方法相比，我们的方案提供了几个优点。首先，整个FL训练课程只需要一个设置阶段，从而显着减少了通信负担。其次，它通过消除用户对用户交互的需要、利用中间服务器层和轻量级密钥协商方法来最大限度地减少用户端的负担。第三，该计划对用户退出具有高度弹性，用户可以在任何FL轮中加入。第四，它可以检测和防御恶意服务器活动，包括最近发现的模型不一致性攻击。最后，我们的方案确保了半诚实和恶意设置中的安全性。我们提供安全分析来正式证明我们方法的稳健性。此外，我们还实现了我们计划的端到端原型。我们进行了全面的实验和比较，结果表明它在通信和计算负担、功能性和安全性方面优于现有解决方案。



## **31. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

通过动态最大化优化改进普遍对抗扰动的推广 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2503.12793v3) [paper-pdf](http://arxiv.org/pdf/2503.12793v3)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.

摘要: 深度神经网络（DNN）容易受到普遍对抗性扰动（UPC）的影响。这些扰动经过精心设计，旨在在所有样本类中普遍欺骗目标模型。与特定于实例的对抗示例（AE）不同，生成UPC更加复杂，因为它们必须在广泛的数据样本和模型中进行概括。我们的研究表明，现有的通用攻击方法使用DNN和静态模型参数快照来优化UAP，但并没有充分利用DNN的潜力来生成更有效的UAP。我们建议使用动态模型-数据对来生成UAP，而不是针对具有固定训练集的静态DNN模型来优化UAP。特别是，我们引入了动态最大值优化策略，旨在在各种最佳模型-数据对上优化UAP。我们将这种方法称为DM-UAP。DM-UAP利用迭代的最大-最小-最小优化框架来细化模型-数据对，并结合课程UAP学习算法来彻底检查模型参数和数据的组合空间。ImageNet数据集的全面实验表明，提出的DM-UAP显着增强了UAP的跨样本通用性和跨模型可移植性。DM-UAP仅使用500个样本即可生成UAP，优于最先进的方法，愚弄率平均增加12.108%。



## **32. SemDiff: Generating Natural Unrestricted Adversarial Examples via Semantic Attributes Optimization in Diffusion Models**

SemDiff：通过扩散模型中的语义属性优化生成自然无限制对抗示例 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11923v1) [paper-pdf](http://arxiv.org/pdf/2504.11923v1)

**Authors**: Zeyu Dai, Shengcai Liu, Rui He, Jiahao Wu, Ning Lu, Wenqi Fan, Qing Li, Ke Tang

**Abstract**: Unrestricted adversarial examples (UAEs), allow the attacker to create non-constrained adversarial examples without given clean samples, posing a severe threat to the safety of deep learning models. Recent works utilize diffusion models to generate UAEs. However, these UAEs often lack naturalness and imperceptibility due to simply optimizing in intermediate latent noises. In light of this, we propose SemDiff, a novel unrestricted adversarial attack that explores the semantic latent space of diffusion models for meaningful attributes, and devises a multi-attributes optimization approach to ensure attack success while maintaining the naturalness and imperceptibility of generated UAEs. We perform extensive experiments on four tasks on three high-resolution datasets, including CelebA-HQ, AFHQ and ImageNet. The results demonstrate that SemDiff outperforms state-of-the-art methods in terms of attack success rate and imperceptibility. The generated UAEs are natural and exhibit semantically meaningful changes, in accord with the attributes' weights. In addition, SemDiff is found capable of evading different defenses, which further validates its effectiveness and threatening.

摘要: 无限制的对抗示例（UAE）允许攻击者在没有给出干净样本的情况下创建无限制的对抗示例，对深度学习模型的安全性构成严重威胁。最近的作品利用扩散模型来生成UAE。然而，由于简单地在中间潜在噪音中进行优化，这些UAE往往缺乏自然性和不可感知性。有鉴于此，我们提出了SemDiff，这是一种新型的无限制对抗攻击，它探索有意义属性的扩散模型的语义潜在空间，并设计了一种多属性优化方法，以确保攻击成功，同时保持生成的UAE的自然性和不可感知性。我们在三个高分辨率数据集（包括CelebA-HQ、AFHQ和ImageNet）上对四项任务进行了广泛的实验。结果表明，SemDiff在攻击成功率和不可感知性方面优于最先进的方法。生成的UAE是自然的，并根据属性的权重表现出有意义的语义变化。此外，SemDiff被发现能够规避不同的防御，这进一步验证了其有效性和威胁性。



## **33. RAB$^2$-DEF: Dynamic and explainable defense against adversarial attacks in Federated Learning to fair poor clients**

RAB $' 2 $-DEF：针对联邦学习中的对抗攻击的动态且可解释的防御，以应对贫困客户 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.08244v2) [paper-pdf](http://arxiv.org/pdf/2410.08244v2)

**Authors**: Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: At the same time that artificial intelligence is becoming popular, concern and the need for regulation is growing, including among other requirements the data privacy. In this context, Federated Learning is proposed as a solution to data privacy concerns derived from different source data scenarios due to its distributed learning. The defense mechanisms proposed in literature are just focused on defending against adversarial attacks and the performance, leaving aside other important qualities such as explainability, fairness to poor quality clients, dynamism in terms of attacks configuration and generality in terms of being resilient against different kinds of attacks. In this work, we propose RAB$^2$-DEF, a $\textbf{r}$esilient $\textbf{a}$gainst $\textbf{b}\text{yzantine}$ and $\textbf{b}$ackdoor attacks which is $\textbf{d}$ynamic, $\textbf{e}$xplainable and $\textbf{f}$air to poor clients using local linear explanations. We test the performance of RAB$^2$-DEF in image datasets and both byzantine and backdoor attacks considering the state-of-the-art defenses and achieve that RAB$^2$-DEF is a proper defense at the same time that it boosts the other qualities towards trustworthy artificial intelligence.

摘要: 在人工智能日益流行的同时，人们对监管的担忧和需求也在增长，其中包括数据隐私等要求。在此背景下，联邦学习因其分布式学习而被提出作为解决不同源数据场景中产生的数据隐私问题的解决方案。文献中提出的防御机制只是专注于防御对抗性攻击和性能，而忽略了其他重要的品质，例如可解释性、对质量较差的客户端的公平性、攻击配置方面的动态性以及针对不同类型攻击的韧性的一般性。在这项工作中，我们提出了RAB $' 2 $-DEF、a $\textBF{r}$esilient $\textBF{a}$gð $\textBF{b}\text{yzantine}$和$\textBF{b}$ackdoor攻击，这是$\textBF{d}$Thomic、$\textBF{e}$xplanable和$\textBF{f}$air使用本地线性解释对贫困客户端进行的攻击。我们测试了RAB $^2 $-DEF在图像数据集和拜占庭攻击和后门攻击中的性能，并考虑到最先进的防御方法，得出RAB $^2 $-DEF是一种适当的防御方法，同时它提高了其他质量，使人工智能变得可信。



## **34. Support is All You Need for Certified VAE Training**

支持就是您认证VAE培训所需的一切 cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11831v1) [paper-pdf](http://arxiv.org/pdf/2504.11831v1)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.

摘要: 变分自动编码器（VAE）变得越来越受欢迎，并部署在安全关键应用中。在此类应用程序中，我们希望为对抗性攻击下的性能提供经过认证的概率保证。我们提出了一种新的方法CIVET，用于VAE的认证培训。CIVET取决于一个关键见解，即我们可以通过将误差限制在潜在层精心选择的支持集上来限制最坏情况的VAE误差。我们以数学方式展示了这一点，并利用这一见解提出了一种新颖的训练算法。我们在对不同数据集（无线和视觉应用领域）、架构和扰动幅度的广泛评估中表明，我们的方法优于SOTA方法，可以在强大的稳健性保证下实现良好的标准性能。



## **35. PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection**

PSBD：预测转变不确定性解锁后门检测 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.05826v2) [paper-pdf](http://arxiv.org/pdf/2406.05826v2)

**Authors**: Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang

**Abstract**: Deep neural networks are susceptible to backdoor attacks, where adversaries manipulate model predictions by inserting malicious samples into the training data. Currently, there is still a significant challenge in identifying suspicious training data to unveil potential backdoor samples. In this paper, we propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. PSBD is motivated by an intriguing Prediction Shift (PS) phenomenon, where poisoned models' predictions on clean data often shift away from true labels towards certain other labels with dropout applied during inference, while backdoor samples exhibit less PS. We hypothesize PS results from the neuron bias effect, making neurons favor features of certain classes. PSBD identifies backdoor training samples by computing the Prediction Shift Uncertainty (PSU), the variance in probability values when dropout layers are toggled on and off during model inference. Extensive experiments have been conducted to verify the effectiveness and efficiency of PSBD, which achieves state-of-the-art results among mainstream detection methods. The code is available at https://github.com/WL-619/PSBD.

摘要: 深度神经网络容易受到后门攻击，攻击者通过将恶意样本插入训练数据来操纵模型预测。目前，在识别可疑的训练数据以揭示潜在的后门样本方面仍然存在重大挑战。在本文中，我们提出了一种新的方法，预测移位后门检测（PSBD），利用基于不确定性的方法，需要最少的未标记的干净的验证数据。PSBD的动机是一个有趣的预测偏移（PS）现象，其中中毒模型对干净数据的预测经常从真实标签转向某些其他标签，在推理过程中应用dropout，而后门样本表现出较少的PS。我们假设PS的结果是神经元偏见效应，使神经元更喜欢某些类别的特征。PSBD通过计算预测漂移不确定性（PSO）来识别后门训练样本，即模型推理期间打开和关闭丢失层时的概率值的方差。人们进行了大量的实验来验证PSBD的有效性和效率，它取得了主流检测方法中最先进的结果。该代码可在https://github.com/WL-619/PSBD上获取。



## **36. Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset**

迈向网络上安全合成图像生成：多模式稳健的NSFW防御和百万规模数据集 cs.CV

Short Paper The Web Conference

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11707v1) [paper-pdf](http://arxiv.org/pdf/2504.11707v1)

**Authors**: Muhammad Shahid Muneer, Simon S. Woo

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: https://github.com/shahidmuneer/multimodal-nsfw-defense.

摘要: 在过去的几年里，我们见证了文本到图像（T2 I）模型的显着成功及其在网络上的广泛使用。对使T2 I模型产生超真实图像的广泛研究引发了新的担忧，例如生成不安全工作（NSFW）网络内容和污染网络社会。为了帮助防止滥用T2 I模型并为用户创建更安全的网络环境，这些模型中使用了NSFW过滤器和事后安全检查等功能。然而，最近的工作揭示了这些方法如何很容易无法防止滥用。特别是，对文本和图像模式的对抗攻击很容易胜过防御措施。%利用这一点导致人们越来越担心防止对文本和图像模式的对抗性攻击。此外，目前还没有包括提示和图像对以及对抗性示例的稳健多模式NSFW数据集。这项工作提出了使用开源扩散模型生成的百万级提示和图像数据集。其次，我们开发了一种多模式防御来区分安全和NSFW文本和图像，该防御系统能够强大地抵御对抗攻击，并直接缓解当前的挑战。我们广泛的实验表明，我们的模型在准确性和召回率方面相对于现有的SOTA NSFW检测方法表现良好，大幅降低了多模式对抗性攻击场景中的攻击成功率（ASB）。代码：https://github.com/shahidmuneer/multimodal-nsfw-defense。



## **37. Learning to Learn Transferable Generative Attack for Person Re-Identification**

学习学习用于人员重新识别的可转移生成攻击 cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2409.04208v2) [paper-pdf](http://arxiv.org/pdf/2409.04208v2)

**Authors**: Yuan Bian, Min Liu, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning-based person re-identification (re-id) models are widely employed in surveillance systems and inevitably inherit the vulnerability of deep networks to adversarial attacks. Existing attacks merely consider cross-dataset and cross-model transferability, ignoring the cross-test capability to perturb models trained in different domains. To powerfully examine the robustness of real-world re-id models, the Meta Transferable Generative Attack (MTGA) method is proposed, which adopts meta-learning optimization to promote the generative attacker producing highly transferable adversarial examples by learning comprehensively simulated transfer-based cross-model\&dataset\&test black-box meta attack tasks. Specifically, cross-model\&dataset black-box attack tasks are first mimicked by selecting different re-id models and datasets for meta-train and meta-test attack processes. As different models may focus on different feature regions, the Perturbation Random Erasing module is further devised to prevent the attacker from learning to only corrupt model-specific features. To boost the attacker learning to possess cross-test transferability, the Normalization Mix strategy is introduced to imitate diverse feature embedding spaces by mixing multi-domain statistics of target models. Extensive experiments show the superiority of MTGA, especially in cross-model\&dataset and cross-model\&dataset\&test attacks, our MTGA outperforms the SOTA methods by 21.5\% and 11.3\% on mean mAP drop rate, respectively. The code of MTGA will be released after the paper is accepted.

摘要: 基于深度学习的人员重新识别（re-id）模型被广泛应用于监控系统中，并且不可避免地继承了深度网络对抗攻击的脆弱性。现有的攻击只考虑了跨数据集和跨模型的可移植性，忽略了交叉测试对不同领域训练的模型的干扰能力。为了有力地检查现实世界re-id模型的稳健性，提出了Meta可转移生成攻击（MTGA）方法，该方法采用元学习优化，通过全面学习模拟的基于转移的交叉模型\&数据集\&测试黑匣子Meta攻击任务，促进生成攻击者产生高度可转移的对抗性示例。具体来说，首先通过为元训练和元测试攻击过程选择不同的re-id模型和数据集来模拟跨模型和数据集的黑匣子攻击任务。由于不同的模型可能关注不同的特征区域，因此进一步设计了扰动随机擦除模块，以防止攻击者学习仅破坏特定于模型的特征。为了促进攻击者学习具有交叉测试可移植性，引入了规范化混合策略，通过混合目标模型的多域统计数据来模仿不同的特征嵌入空间。大量实验表明了MTGA的优越性，特别是在跨模型数据集和跨模型数据集测试攻击中，我们的MTGA在平均mAP下降率方面分别比SOTA方法高出21.5%和11.3%。论文被接受后，MTGA的代码将发布。



## **38. Propaganda via AI? A Study on Semantic Backdoors in Large Language Models**

通过人工智能进行宣传？大型语言模型中的语义后门研究 cs.CL

18 pages, 1 figure

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.12344v1) [paper-pdf](http://arxiv.org/pdf/2504.12344v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at https://github.com/NayMyatMin/RAVEN.

摘要: 大型语言模型（LLM）在无数语言任务中表现出出色的性能，但它们仍然容易受到后门攻击，即对手植入隐藏触发器来系统性地操纵模型输出。传统防御专注于显式标记级异常，因此忽视了嵌入在概念级的语义后门隐蔽触发器（例如，意识形态立场或文化参考）依赖于基于意义的线索，而不是词汇上的怪异。我们首先表明，在受控微调环境中，这种语义后门只能植入一个小的有毒主体，从而建立了它们的实际可行性。然后，我们在LLM中形式化了语义后门的概念，并引入了黑匣子检测框架RAVEN（“揭露语义后门的响应异常警戒”的缩写），该框架将语义熵与跨模型一致性分析相结合。该框架通过结构化的主题视角提示来探索多个模型，通过双向蕴含对采样的响应进行聚集，并标记出极其均匀的输出;跨模型比较将模型特定的异常与整个群体的偏差隔离开来。对不同LLM家族（GPT-4 o、Llama、DeepSeek、Mistral）的经验评估揭示了之前未检测到的语义后门，提供了这些隐藏漏洞的第一个概念验证证据，并强调了对已部署语言模型进行概念级审计的迫切需要。我们在https://github.com/NayMyatMin/RAVEN上开源我们的代码和数据。



## **39. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）在整个过程中提供语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。本文的研究可以进一步引起人们对多媒体信息安全的关注。



## **40. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

显而易见的不可见威胁：LLM-Powered GUI代理对Fine-Print注入的脆弱性 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.

摘要: 由大型语言模型（LLM）驱动的图形用户界面代理是一个专门的自治系统，根据高级指令代表用户执行任务。它通过感知和解释相关应用程序的图形用户界面（GUIs）（通常是视觉上的），推断必要的操作序列，然后通过执行单击、打字和点击等操作与GUIs交互来实现这一目标。为了完成现实世界的任务，例如填写表格或预订服务，图形用户界面代理通常需要处理和处理敏感用户数据。然而，这种自主性带来了新的隐私和安全风险。对手可以将恶意内容注入图形用户界面，从而改变代理行为或导致私人信息的意外泄露。这些攻击通常利用代理和人类用户的视觉显著性之间的差异，或者代理检测任务自动化中上下文完整性违规的能力有限。在本文中，我们描述了六种类型的此类攻击，并进行了一项实验研究，使用六个最先进的图形用户界面代理、234个对抗性网页和39名人类参与者来测试这些攻击。我们的研究结果表明，图形用户界面代理非常容易受到攻击，特别是对于上下文嵌入式威胁。此外，人类用户也容易受到许多此类攻击，这表明简单的人类监督可能无法可靠地防止故障。这种错位凸显了隐私感知代理设计的必要性。我们提出了实用的防御策略，为开发更安全、更可靠的图形用户界面代理提供信息。



## **41. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

Slice+Slice Baby：眨眼间生成末级缓存驱逐集 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11208v1) [paper-pdf](http://arxiv.org/pdf/2504.11208v1)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.

摘要: 发起缓存攻击的一个重要步骤是找到驱逐集，即争夺缓存空间的内存位置集合。在英特尔处理器上，识别竞争地址的主要挑战之一是切片高速缓存设计，其中处理器对物理地址进行哈希处理以确定内存位置存储在高速缓存中的位置。虽然过去的工作已经证明哈希函数可以颠倒，但他们也表明它取决于对手不知道的物理地址位。   在这部作品中，我们对寻找驱逐集的艺术做出了三项主要贡献。我们首先利用微体系结构的竞争比较内存访问时间，并确定地址映射到的缓存片。然后，我们使用已知的散列函数，以减少我们的切片识别方法中的错误率，并通过将切片映射外推到未经测试的内存地址来减少工作。最后，我们展示了如何传播信息驱逐集在不同的页面偏移量的非线性散列函数的情况下，迄今为止尚未探索。   我们的贡献允许在Intel i7- 9850 H上在0.7秒内生成整个LLC驱逐集，在i9- 10900 K上在1.6秒内生成整个LLC驱逐集，两者都使用非线性函数。与最先进的技术相比，这是一个显着的改进，分别耗时9倍和10倍。



## **42. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

R-TPT：通过测试时提示调优提高视觉语言模型的对抗鲁棒性 cs.LG

CVPR 2025

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11195v1) [paper-pdf](http://arxiv.org/pdf/2504.11195v1)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.

摘要: CLIP等视觉语言模型（VLM）作为基础模型已受到广泛欢迎，并开发了多种微调方法来增强下游任务的性能。然而，由于其固有的脆弱性以及从有限的开源模型集中进行选择的常见做法，VLM比传统视觉模型面临更高的对抗攻击风险。现有的防御技术通常依赖于训练期间的对抗微调，这需要标记数据并且缺乏下游任务的灵活性。为了解决这些限制，我们提出了鲁棒的测试时即时调优（R-TPT），它可以减轻推理阶段对抗性攻击的影响。我们首先通过消除在对抗条件下引入冲突的术语来重新制定经典的边际熵目标，只保留逐点的熵最小化。此外，我们引入了一种即插即用的、基于可靠性的加权集成策略，该策略从可靠的增强视图中聚合有用信息以加强防御。R-TPT增强了对对抗攻击的防御，而不需要标记的训练数据，同时为推理任务提供高度灵活性。对广泛使用的具有各种攻击的基准进行了大量实验，证明了R-TPT的有效性。该代码可在https://github.com/TomSheng21/R-TPT上找到。



## **43. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

探索LLM授权建议的后门攻击和防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.

摘要: 大型语言模型（LLM）与推荐系统（RecSys）的融合极大地提高了个性化推荐并引起了广泛关注。尽管取得了令人印象深刻的进展，但基于LLM的RecSys抵御后门攻击的安全性在很大程度上仍然没有得到充分的探索。在本文中，我们提出了一个新问题：具有特定触发器的后门是否会被注入到基于LLM的Recsys中，从而导致当后门触发器附加到项目标题时推荐响应的操纵？为了调查基于LLM的RecSys在后门攻击下的漏洞，我们提出了一种新的攻击框架，称为RecSys后门注入中毒（BadRec）。BadRec通过触发器扰乱这些物品的标题，并雇用几名虚假用户与这些物品互动，有效地毒害了训练集，并为基于LLM的RecSys注入后门。全面的实验表明，仅用对抗性示例毒害1%的训练数据就足以成功植入后门，从而能够操纵推荐。为了进一步减轻此类安全威胁，我们提出了一种名为毒药扫描仪（P-Scanner）的通用防御策略。具体来说，我们引入了基于LLM的毒物扫描仪，通过利用LLM强大的语言理解能力和丰富的知识来检测有毒物品。触发增强代理被用来生成不同的合成触发器，以引导中毒扫描器学习中毒物品检测任务的特定于领域的知识。在三个真实数据集上的大量实验验证了所提出的P-Scanner的有效性。



## **44. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

安全的文本到图像生成：简单地消除提示嵌入 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.10329v2) [paper-pdf](http://arxiv.org/pdf/2411.10329v2)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to some extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a core requirement of T2I models, we identify that textual representations are likely the primary source of unsafe generation. To this end, we propose Embedding Sanitizer (ES), which enhances the safety of T2I models by sanitizing inappropriate concepts in prompt embeddings. To our knowledge, ES is the first interpretable safe generation framework that assigns a score to each token in the prompt to indicate its potential harmfulness. In addition, ES adopts a plug-and-play modular design, offering compatibility for seamless integration with various T2I models and other safeguards. Evaluations on five prompt benchmarks show that ES outperforms eleven existing safeguard baselines, achieving state-of-the-art robustness while maintaining high-quality image generation.

摘要: 近年来，文本到图像（T2 I）生成模型在生成与文本描述一致的高质量图像方面取得了重大进展。然而，这些模型还面临着不安全生成的风险，可能会产生违反使用政策的有害内容，例如显式材料。现有的安全生成方法通常专注于通过从视觉表示中删除不需要的概念来抑制不适当的内容，同时忽视对文本表示的净化。尽管这些方法在一定程度上有助于降低滥用的风险，但在处理对抗性攻击时，它们的稳健性仍然不足。   鉴于输入文本和输出图像之间的语义一致性是T2 I模型的核心要求，我们发现文本表示可能是不安全生成的主要来源。为此，我们提出了嵌入Sanitizer（ES），它通过在提示嵌入中清理不适当的概念来增强T2 I模型的安全性。据我们所知，ES是第一个可解释的安全生成框架，它为提示中的每个令牌分配一个分数，以指示其潜在的危害性。此外，ES采用即插即用的模块化设计，提供与各种T2 I型号和其他保障措施的无缝集成的兼容性。对五个即时基准的评估表明，ES优于十一个现有的保障基线，实现了最先进的鲁棒性，同时保持高质量的图像生成。



## **45. Helper-Friendly Latency-Bounded Mitigation Strategies against Reactive Jamming Adversaries**

针对反应性干扰对手的助手友好的、有延迟限制的缓解策略 cs.IT

16 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11110v1) [paper-pdf](http://arxiv.org/pdf/2504.11110v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: Due to the recent developments in the field of full-duplex radios and cognitive radios, a new class of reactive jamming attacks has gained attention wherein an adversary transmits jamming energy over the victim's frequency band and also monitors various energy statistics in the network so as to detect countermeasures, thereby trapping the victim. Although cooperative mitigation strategies against such security threats exist, they are known to incur spectral-efficiency loss on the helper node, and are also not robust to variable latency-constraints on victim's messages. Identifying these research gaps in existing countermeasures against reactive jamming attacks, we propose a family of helper-friendly cooperative mitigation strategies that are applicable for a wide-range of latency-requirements on the victim's messages as well as practical radio hardware at the helper nodes. The proposed strategies are designed to facilitate reliable communication for the victim, without compromising the helper's spectral efficiency and also minimally disturbing the various energy statistics in the network. For theoretical guarantees on their efficacy, interesting optimization problems are formulated on the choice of the underlying parameters, followed by extensive mathematical analyses on their error-performance and covertness. Experimental results indicate that the proposed strategies should be preferred over the state-of-the-art methods when the helper node is unwilling to compromise on its error performance for assisting the victim.

摘要: 由于目前在全频无线电和认知无线电领域的发展，一类新型反应性干扰攻击引起了人们的关注，其中对手在受害者的频段上传输干扰能量，并监控网络中的各种能量统计数据，以检测对抗措施，从而陷阱受害者。尽管存在针对此类安全威胁的合作缓解策略，但众所周知，它们会在帮助节点上导致频谱效率损失，并且对受害者消息的可变延迟约束也不鲁棒。通过识别针对反应性干扰攻击的现有对策中的这些研究差距，我们提出了一系列对助手友好的合作缓解策略，这些策略适用于受害者消息的广泛延迟要求以及助手节点处的实用无线电硬件。所提出的策略旨在促进受害者的可靠通信，而不损害助手的频谱效率，并且最大限度地干扰网络中的各种能量统计数据。为了从理论上保证其功效，在基本参数的选择上制定了有趣的优化问题，然后对其错误性能和隐蔽性进行了广泛的数学分析。实验结果表明，当帮助节点不愿意牺牲其帮助受害者的错误性能时，所提出的策略应该优于最先进的方法。



## **46. Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models**

越狱文本到图像模型的令牌级约束边界搜索 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11106v1) [paper-pdf](http://arxiv.org/pdf/2504.11106v1)

**Authors**: Jiangtao Liu, Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Recent advancements in Text-to-Image (T2I) generation have significantly enhanced the realism and creativity of generated images. However, such powerful generative capabilities pose risks related to the production of inappropriate or harmful content. Existing defense mechanisms, including prompt checkers and post-hoc image checkers, are vulnerable to sophisticated adversarial attacks. In this work, we propose TCBS-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. By iteratively optimizing tokens near these boundaries, TCBS-Attack generates semantically coherent adversarial prompts capable of bypassing multiple defensive layers in T2I models. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 45\% and an ASR-1 of 21\% on jailbreaking full-chain T2I models, significantly surpassing baseline methods.

摘要: 文本到图像（T2 I）生成的最新进展显着增强了生成图像的真实感和创造力。然而，如此强大的生成能力带来了与制作不适当或有害内容相关的风险。现有的防御机制，包括提示检查器和事后图像检查器，很容易受到复杂的对抗攻击。在这项工作中，我们提出了TCBS-Attack，这是一种新型的基于查询的黑匣子越狱攻击，可以搜索位于文本和图像检查器定义的决策边界附近的令牌。通过迭代优化这些边界附近的令牌，TCBS-Attack生成语义一致的对抗性提示，能够绕过T2 I模型中的多个防御层。大量的实验表明，我们的方法在各种T2 I模型中始终优于最先进的越狱攻击，包括安全训练的开源模型和商业在线服务，如DALL-E 3。TCBS-Attack在越狱全链T2 I模型上实现了45%的ASR-4和21%的ASR-1，大大超过了基线方法。



## **47. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

揭露欺诈团伙对图神经网络的威胁：针对基于GNN的欺诈检测器的多目标图注入攻击 cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.18370v3) [paper-pdf](http://arxiv.org/pdf/2412.18370v3)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.

摘要: 图神经网络（GNN）已经成为欺诈检测、识别欺诈用户和发现恶意行为的有效工具。然而，对基于GNN的欺诈检测器的攻击及其风险很少被研究，从而使潜在的威胁未得到解决。最近的调查结果表明，欺诈越来越多地以团伙或集团的形式组织起来。在这项工作中，我们设计的攻击场景中，欺诈团伙的目标是使他们的欺诈节点被错误地归类为良性的勾结，通过掩盖他们的非法活动。基于这些场景，我们通过模拟三个现实世界的欺诈案件（垃圾邮件评论、假新闻和医疗保险欺诈）中欺诈团伙的攻击，研究针对基于GNN的欺诈检测器的对抗攻击。我们将这些攻击定义为多目标图注入攻击，并提出MonTi，一种基于变换器的多目标一次性图注入攻击模型。MonTi通过Transformer编码器同时生成所有攻击节点的属性和边，比大多数按顺序生成这些元素的现有图注入攻击方法更有效地捕获属性和边之间的相互依赖性。此外，MonTi自适应地为每个攻击节点分配度预算，以探索涉及目标、候选和攻击节点的不同注入结构，而不像现有方法那样固定所有攻击节点的度预算。实验表明，MonTi在五个现实世界的图上的表现优于最先进的图注入攻击方法。



## **48. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIIR：基于互信息的对抗鲁棒性的掩蔽图像建模 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2312.04960v4) [paper-pdf](http://arxiv.org/pdf/2312.04960v4)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism.

摘要: 视觉变形者（ViT）已成为一种基本架构，并成为现代视觉语言模型的支柱。尽管ViT的性能令人印象深刻，但其对规避攻击表现出明显的脆弱性，因此需要开发针对其独特架构定制的专门对抗训练（AT）策略。虽然直接的解决方案可能涉及将现有的AT方法应用于ViT，但我们的分析揭示了显着的不兼容性，特别是与最先进的（SOTA）方法，例如Generalist（CVPR 2023）和DBAT（USENIX Security 2024）。本文对ViT中的对抗鲁棒性进行了系统研究，并在其基于自动编码器的自我监督预训练中提供了一种新颖的理论互信息（MI）分析。具体来说，我们表明，对抗性的例子和它的潜在的表示在基于ViT的自动编码器之间的MI应通过派生的MI边界的约束。基于这一见解，我们提出了一种自监督AT方法MIMIR，该方法采用MI惩罚，通过使用自动编码器进行掩码图像建模来促进对抗性预训练。在CIFAR-10、Tiny-ImageNet和ImageNet-1 K上的大量实验表明，MIMIR可以始终如一地提供更好的自然和鲁棒的准确性，其中MIMIR在ImageNet-1 K上的结果优于SOTA AT。值得注意的是，MIMIR对不可预见的攻击和常见的损坏数据表现出卓越的鲁棒性，并且还可以承受对手完全了解防御机制的自适应攻击。



## **49. RAID: An In-Training Defense against Attribute Inference Attacks in Recommender Systems**

RAGE：推荐系统中针对属性推理攻击的训练中防御 cs.IR

17 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11510v1) [paper-pdf](http://arxiv.org/pdf/2504.11510v1)

**Authors**: Xiaohua Feng, Yuyuan Li, Fengyuan Yu, Ke Xiong, Junjie Fang, Li Zhang, Tianyu Du, Chaochao Chen

**Abstract**: In various networks and mobile applications, users are highly susceptible to attribute inference attacks, with particularly prevalent occurrences in recommender systems. Attackers exploit partially exposed user profiles in recommendation models, such as user embeddings, to infer private attributes of target users, such as gender and political views. The goal of defenders is to mitigate the effectiveness of these attacks while maintaining recommendation performance. Most existing defense methods, such as differential privacy and attribute unlearning, focus on post-training settings, which limits their capability of utilizing training data to preserve recommendation performance. Although adversarial training extends defenses to in-training settings, it often struggles with convergence due to unstable training processes. In this paper, we propose RAID, an in-training defense method against attribute inference attacks in recommender systems. In addition to the recommendation objective, we define a defensive objective to ensure that the distribution of protected attributes becomes independent of class labels, making users indistinguishable from attribute inference attacks. Specifically, this defensive objective aims to solve a constrained Wasserstein barycenter problem to identify the centroid distribution that makes the attribute indistinguishable while complying with recommendation performance constraints. To optimize our proposed objective, we use optimal transport to align users with the centroid distribution. We conduct extensive experiments on four real-world datasets to evaluate RAID. The experimental results validate the effectiveness of RAID and demonstrate its significant superiority over existing methods in multiple aspects.

摘要: 在各种网络和移动应用程序中，用户极易受到属性推断攻击，其中推荐系统中尤其普遍。攻击者利用推荐模型中部分暴露的用户配置文件（例如用户嵌入）来推断目标用户的私人属性，例如性别和政治观点。防御者的目标是降低这些攻击的有效性，同时保持推荐性能。大多数现有的防御方法，例如差异隐私和属性取消学习，都集中在训练后设置上，这限制了它们利用训练数据来保持推荐性能的能力。尽管对抗性训练将防御扩展到训练中的环境，但由于训练过程不稳定，它经常难以收敛。在本文中，我们提出了一种针对推荐系统中属性推理攻击的训练中防御方法--RAIDER。除了推荐目标之外，我们还定义了一个防御目标，以确保受保护属性的分布独立于类标签，使用户无法与属性推断攻击区分开来。具体来说，这个防御目标旨在解决受约束的Wasserstein重心问题，以识别使属性不可区分的重心分布，同时遵守推荐性能约束。为了优化我们提出的目标，我们使用最佳传输来使用户与重心分布保持一致。我们对四个现实世界的数据集进行了广泛的实验来评估磁盘阵列。实验结果验证了磁盘阵列的有效性，并展示了其在多个方面相对于现有方法的显着优势。



## **50. Inferring Communities of Interest in Collaborative Learning-based Recommender Systems**

基于协作学习的推荐系统中的兴趣社区推断 cs.IR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2306.08929v3) [paper-pdf](http://arxiv.org/pdf/2306.08929v3)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems, such as those employing Federated Learning (FL) and Gossip Learning (GL), allow users to train models while keeping their history of liked items on their devices. While these methods were seen as promising for enhancing privacy, recent research has shown that collaborative learning can be vulnerable to various privacy attacks. In this paper, we propose a novel attack called Community Inference Attack (CIA), which enables an adversary to identify community members based on a set of target items. What sets CIA apart is its efficiency: it operates at low computational cost by eliminating the need for training surrogate models. Instead, it uses a comparison-based approach, inferring sensitive information by comparing users' models rather than targeting any specific individual model. To evaluate the effectiveness of CIA, we conduct experiments on three real-world recommendation datasets using two recommendation models under both Federated and Gossip-like settings. The results demonstrate that CIA can be up to 10 times more accurate than random guessing. Additionally, we evaluate two mitigation strategies: Differentially Private Stochastic Gradient Descent (DP-SGD) and a Share less policy, which involves sharing fewer, less sensitive model parameters. Our findings suggest that the Share less strategy offers a better privacy-utility trade-off, especially in GL.

摘要: 基于协作学习的推荐系统，例如采用联邦学习（FL）和Gossip Learning（GL）的推荐系统，允许用户训练模型，同时在设备上保留喜欢项目的历史记录。虽然这些方法被认为有希望增强隐私，但最近的研究表明，协作学习可能容易受到各种隐私攻击。在本文中，我们提出了一种名为社区推理攻击（CIA）的新型攻击，它使对手能够根据一组目标项识别社区成员。CIA的与众不同之处在于其效率：它通过消除训练代理模型的需要，以较低的计算成本运行。相反，它使用基于比较的方法，通过比较用户的模型而不是针对任何特定的个体模型来推断敏感信息。为了评估CIA的有效性，我们在联邦和类似Gossip的环境下使用两种推荐模型对三个现实世界的推荐数据集进行了实验。结果表明，CIA的准确性比随机猜测高出10倍。此外，我们还评估了两种缓解策略：差异私人随机梯度下降（DP-BCD）和共享较少政策，该政策涉及共享更少、更不敏感的模型参数。我们的研究结果表明，少分享策略提供了更好的隐私与公用事业权衡，尤其是在GL中。



