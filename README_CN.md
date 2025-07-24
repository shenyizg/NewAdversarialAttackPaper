# Latest Adversarial Attack Papers
**update at 2025-07-24 10:59:36**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Constructing Optimal Noise Channels for Enhanced Robustness in Quantum Machine Learning**

构建最佳噪音通道以增强量子机器学习的鲁棒性 quant-ph

QML technical track at IEEE QCE 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2404.16417v2) [paper-pdf](http://arxiv.org/pdf/2404.16417v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: With the rapid advancement of Quantum Machine Learning (QML), the critical need to enhance security measures against adversarial attacks and protect QML models becomes increasingly evident. In this work, we outline the connection between quantum noise channels and differential privacy (DP), by constructing a family of noise channels which are inherently $\epsilon$-DP: $(\alpha, \gamma)$-channels. Through this approach, we successfully replicate the $\epsilon$-DP bounds observed for depolarizing and random rotation channels, thereby affirming the broad generality of our framework. Additionally, we use a semi-definite program to construct an optimally robust channel. In a small-scale experimental evaluation, we demonstrate the benefits of using our optimal noise channel over depolarizing noise, particularly in enhancing adversarial accuracy. Moreover, we assess how the variables $\alpha$ and $\gamma$ affect the certifiable robustness and investigate how different encoding methods impact the classifier's robustness.

摘要: 随着量子机器学习（QML）的快速发展，增强对抗性攻击的安全措施和保护QML模型的迫切需要变得越来越明显。在这项工作中，我们概述了量子噪声通道和差分隐私（DP）之间的联系，通过构建一个家庭的噪声通道，这是固有的$\alpha，\gamma $-DP：$（\alpha，\gamma）$-通道。通过这种方法，我们成功地复制了$\N $-DP界观察到的去极化和随机旋转通道，从而肯定了我们的框架的广泛的一般性。此外，我们使用一个半定规划，以构建一个最佳的鲁棒信道。在一个小规模的实验评估中，我们展示了使用我们的最佳噪声通道去极化噪声的好处，特别是在提高对抗精度。此外，我们评估变量$\alpha$和$\gamma$如何影响可证明的鲁棒性，并研究不同的编码方法如何影响分类器的鲁棒性。



## **2. Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors**

增强具有基于传输的先验的硬标签攻击的Ray搜索程序 cs.CV

Published at ICLR 2025 (Spotlight paper)

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17577v1) [paper-pdf](http://arxiv.org/pdf/2507.17577v1)

**Authors**: Chen Ma, Xinjie Xu, Shuyu Cheng, Qi Xuan

**Abstract**: One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.

摘要: 最实用、最具挑战性的黑匣子对抗攻击类型之一是硬标签攻击，其中只有前1名的预测标签可用。一种有效的方法是从良性图像中搜索最佳射线方向，以最小化与对抗区域的$\ell_p$-norm距离。这种方法的独特优势在于，它将硬标签攻击转化为持续优化问题。目标函数值是射线的半径，可以通过二分搜索以很高的查询成本获得。现有方法在梯度估计中使用“符号技巧”来减少查询数量。本文从理论上分析了这种梯度估计的质量，并提出了一种新型的先验引导方法来从理论上和经验上提高射线搜索效率。具体来说，我们利用来自代理模型的基于转移的先验，我们的梯度估计器通过以查询高效的方式将真实梯度的投影逼近到由这些先验和随机方向跨越的子空间上来适当地集成它们。我们从理论上推导出所获得的梯度估计量和真实梯度之间的预期Cosin相似性，并证明通过合并先验来实现的改进。对ImageNet和CIFAR-10数据集的大量实验表明，我们的方法在查询效率方面显着优于11种最先进的方法。



## **3. An h-space Based Adversarial Attack for Protection Against Few-shot Personalization**

一种基于h空间的对抗攻击，用于防止少镜头个性化 cs.CV

32 pages, 15 figures. Accepted by ACM Multimedia 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17554v1) [paper-pdf](http://arxiv.org/pdf/2507.17554v1)

**Authors**: Xide Xu, Sandesh Kamath, Muhammad Atif Butt, Bogdan Raducanu

**Abstract**: The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.

摘要: 扩散模型的多功能性，在生成定制的图像从几个样本提出了显着的隐私问题，特别是关于未经授权的修改私人内容。这一令人关注的问题重新致力于开发基于对抗性攻击的保护机制，这对毒药扩散模型产生了有效的扰动。我们的工作是出于观察，这些模型表现出高度的抽象在其语义的潜在空间（“h-空间”），编码关键的高层次的功能，产生连贯和有意义的内容。在本文中，我们提出了一种新的反定制方法，称为HAAD（基于h空间的对抗性攻击扩散模型），它利用对抗性攻击来制作基于h空间的扰动，可以有效地降低图像生成过程。在HAAD的基础上，我们进一步引入了一种更有效的变体HAAD-KV，它仅根据h空间的KV参数构建扰动。该策略提供了更强的保护，计算成本更低。尽管它们很简单，但我们的方法优于最先进的对抗攻击，凸显了它们的有效性。



## **4. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.

摘要: 对抗性越狱攻击的最新进展暴露了大型语言模型（LLM）中的关键漏洞，从而能够通过日益复杂的提示操作来规避对齐保障措施。根据我们的实验，我们发现越狱策略的有效性受到被攻击LLM理解能力的影响。基于这一见解，我们提出了一个功能感知的多加密框架（MEF），用于评估黑匣子LLM中的漏洞。具体来说，MEF首先对LLM的理解能力水平进行分类，然后相应地应用不同的策略：对于理解能力有限的模型，MEF采用Du + En 1策略，将分层语义突变与加密技术集成在一起，更有效地帮助规避LLM在输入和推理阶段的防御。对于理解能力强的模型，MEF使用更复杂的Fu+ En 1 + En 2策略，其中额外的双端加密技术应用于LLM的响应，进一步有助于在输出阶段逃避LLM的防御。实验结果证明了我们方法的有效性，在GPT-4 o（2025年5月29日发布）和GPT-4.1（2025年7月8日发布）上的攻击成功率分别为98.9%和99.8%。我们的工作有助于更深入地了解当前LLM对齐机制中的漏洞。



## **5. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

使用LLM的显式漏洞生成：对抗性攻击之外的调查 cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.10054v2) [paper-pdf](http://arxiv.org/pdf/2507.10054v2)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.

摘要: 大型语言模型（LLM）越来越多地被用作代码助手，但当被明确要求生成不安全代码时，它们的行为仍然知之甚少。虽然之前的研究重点是无意的漏洞，但这项研究考察了一个更直接的威胁：开源LLM在提示时生成易受攻击的代码。我们提出了一种双重实验设计：（1）动态预算处理，它系统地改变结构化模板中的漏洞类型、用户角色和提示措辞;（2）反向预算处理，它从真正的脆弱代码样本中派生自然语言提示。我们使用静态分析评估了三个开源7 B参数模型（Qwen 2、Mistral、Gemma），以评估生成漏洞的存在性和正确性。我们的结果表明，所有模型都会经常生成请求的漏洞，尽管性能差异很大。Gemma在动态预算分配下实现了最高的内存漏洞正确性（例如，缓冲区溢出为98.6%），而Qwen 2在所有任务中表现出最平衡的性能。我们发现专业角色（例如，“DevOps工程师”）始终比学生角色获得更高的成功率，并且直接措辞与间接措辞的有效性根据提示策略而颠倒。漏洞复制准确性遵循具有代码复杂性的非线性模式，峰值在中等范围内。我们的研究结果揭示了LLM对模式回忆而不是语义推理的依赖如何在其安全性排列中造成了显着的盲点，特别是对于被定义为看似合理的专业任务的请求。



## **6. Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise**

广义相关噪声下分散学习中隐私-效用权衡的优化 cs.LG

6 pages, 5 figures, accepted at IEEE ITW 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2501.14644v2) [paper-pdf](http://arxiv.org/pdf/2501.14644v2)

**Authors**: Angelo Rodio, Zheng Chen, Erik G. Larsson

**Abstract**: Decentralized learning enables distributed agents to collaboratively train a shared machine learning model without a central server, through local computation and peer-to-peer communication. Although each agent retains its dataset locally, sharing local models can still expose private information about the local training datasets to adversaries. To mitigate privacy attacks, a common strategy is to inject random artificial noise at each agent before exchanging local models between neighbors. However, this often leads to utility degradation due to the negative effects of cumulated artificial noise on the learning algorithm. In this work, we introduce CorN-DSGD, a novel covariance-based framework for generating correlated privacy noise across agents, which unifies several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, CorN-DSGD optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that CorN-DSGD cancels more noise than existing pairwise correlation schemes, improving model performance under formal privacy guarantees.

摘要: 去中心化学习使分布式代理能够通过本地计算和点对点通信，在没有中央服务器的情况下协作训练共享机器学习模型。尽管每个代理都在本地保留其数据集，但共享本地模型仍然可能向对手暴露有关本地训练数据集的私人信息。为了减轻隐私攻击，一种常见的策略是在邻居之间交换本地模型之前向每个代理注入随机人工噪音。然而，由于累积人工噪音对学习算法的负面影响，这通常会导致效用下降。在这项工作中，我们引入了CorN-DBCD，这是一种新型的基于协方差的框架，用于在代理之间生成相关隐私噪音，它将多种最先进的方法统一为特例。通过利用网络布局和混合权重，CorN-DBCD优化噪音协方差，以实现网络范围的噪音消除。实验结果表明，CorN-DBCD比现有的成对相关方案消除了更多的噪音，在形式隐私保证下提高了模型性能。



## **7. Restricted Boltzmann machine as a probabilistic Enigma**

作为概率谜的限制Boltzmann机 cond-mat.stat-mech

7 pages, 4 figures

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17236v1) [paper-pdf](http://arxiv.org/pdf/2507.17236v1)

**Authors**: Bin Chen, Weichao Yu

**Abstract**: We theoretically propose a symmetric encryption scheme based on Restricted Boltzmann Machines that functions as a probabilistic Enigma device, encoding information in the marginal distributions of visible states while utilizing bias permutations as cryptographic keys. Theoretical analysis reveals significant advantages including factorial key space growth through permutation matrices, excellent diffusion properties, and computational complexity rooted in sharp P-complete problems that resist quantum attacks. Compatible with emerging probabilistic computing hardware, the scheme establishes an asymmetric computational barrier where legitimate users decrypt efficiently while adversaries face exponential costs. This framework unlocks probabilistic computers' potential for cryptographic systems, offering an emerging encryption paradigm between classical and quantum regimes for post-quantum security.

摘要: 从理论上讲，我们提出了一种基于受限制的Boltzmann机的对称加密方案，该方案充当概率谜设备，以可见状态的边缘分布对信息进行编码，同时利用偏差排列作为密钥。理论分析揭示了显着的优势，包括通过置换矩阵的因乘密钥空间增长、出色的扩散性质以及源于抵抗量子攻击的尖锐P完全问题的计算复杂性。该方案与新兴的概率计算硬件兼容，建立了一个不对称的计算障碍，合法用户可以有效解密，而对手则面临指数级成本。该框架释放了概率计算机在密码系统中的潜力，为后量子安全提供了经典和量子机制之间的新兴加密范式。



## **8. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2502.20650v4) [paper-pdf](http://arxiv.org/pdf/2502.20650v4)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获得。



## **9. Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach**

通过集群防御方法提高深度强化学习的鲁棒性 cs.LG

6 pages, 4 figures, 2 tables

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.17070v1) [paper-pdf](http://arxiv.org/pdf/2507.17070v1)

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies.

摘要: 深度强化学习（DRL）的最新进展已经证明了其在各个领域的适用性，包括机器人、医疗保健、能源优化和自动驾驶。然而，一个关键问题仍然存在：DRL模型在遭受对抗攻击时有多稳健？虽然对抗训练和提炼等现有防御机制增强了DRL模型的弹性，但关于自动驾驶场景中多种防御的集成仍然存在显着的研究空白。本文通过提出一种新颖的基于集成的防御架构来缓解自动驾驶中的对抗攻击来解决这一差距。我们的评估表明，提出的架构显着增强了DRL模型的鲁棒性。与FGSM攻击下的基线相比，我们的集成方法在高速公路场景和合并场景中将平均奖励从5.87提高到18.38（增加超过213%），并将平均碰撞率从0.50降低到0.09（减少82%），优于所有独立防御策略。



## **10. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

当LLM复制到思考时：揭示推理LLM中的复制引导攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.

摘要: 大型语言模型（LLM）已成为自动代码分析的组成部分，可以实现漏洞检测和代码理解等任务。然而，它们的集成引入了新颖的攻击表面。在本文中，我们识别并研究了一类新的基于预算的攻击，称为复制引导攻击（CGA），它利用了具有推理能力的LLM固有的复制倾向。通过将精心制作的触发器注入外部代码片段中，对手可以诱导模型在推理期间复制恶意内容。这种行为导致两类漏洞：推理长度操纵，其中模型生成异常短或过长的推理痕迹;和推理结果操纵，其中模型生成误导性或不正确的结论。我们将CGA形式化为一个优化问题，并提出一种基于梯度的方法来合成有效的触发器。对最先进推理LLM的实证评估表明，CGA可靠地在代码分析任务中引发无限循环、提前终止、错误拒绝和语义扭曲。虽然在目标环境中非常有效，但由于计算限制，我们观察到在不同提示中推广CGA存在挑战，这为未来的研究提出了一个悬而未决的问题。我们的研究结果揭示了LLM驱动的开发管道中一个关键但未充分探索的漏洞，并呼吁在预算级防御机制方面紧急取得进展。



## **11. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

ShadowCode：针对代码LLM的（自动）外部提示注入攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.

摘要: 最近的进步导致面向代码的大型语言模型（Code LLM）被广泛采用来进行编程任务。尽管他们在部署方面取得了成功，但他们的安全研究却远远落后。本文引入了一种新的攻击范式：针对Code LLM的（自动）外部提示注入，攻击者生成简洁的、非功能性的诱导扰动，并将其注入到受害者的代码上下文中。这些引发的扰动可以通过常用的依赖性传播（例如，包或RAG的知识库），在代码完成过程中操纵代码LLM以实现恶意目标。与现有的攻击相比，这种方法更现实、更具威胁性：与后门攻击不同，它不需要控制模型的训练过程，并且可以实现对对抗性攻击具有挑战性的特定恶意目标。此外，我们提出了ShadowCode，这是一种简单而有效的方法，可以根据代码模拟自动生成诱导扰动，以实现有效且隐蔽的外部提示注入。ShadowCode通过模拟现实代码上下文来设计其扰动优化目标，并采用具有两个增强模块的贪婪优化方法：前向推理增强和基于关键字的扰动设计。我们针对13个不同的恶意目标评估我们的方法，生成了跨越三种流行编程语言的31个威胁案例。我们的结果表明，ShadowCode仅使用12个令牌的非功能性诱导扰动，就成功攻击了所有威胁案例中的三个代表性开源Code LLM（攻击成功率高达97.9%）和两个主流商业Code LLM集成应用程序（攻击成功率超过90%）。该代码可在https://github.com/LianPing-cyber/ShadowCodeEPI上获取。



## **12. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

压缩的成本：对草图进行严格的二次黑匣子攻击，以获取$\ell_2 $ Norm估计 cs.LG

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16345v1) [paper-pdf](http://arxiv.org/pdf/2507.16345v1)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix A in $R^{k X n}$ maps high-dimensional vectors v $\in R^n$ to lower-dimensional sketches A v in $R^k$, and an adversary can query the system to obtain approximate ell2-norm estimates that are computed from the sketch.   We present a universal, nonadaptive attack that, using tilde(O)($k^2$) queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: It applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution.   Our lower bound construction tightly matches the known upper bounds of tilde(Omega)($k^2$), achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.

摘要: 通过线性草图减少维度是一种强大且广泛使用的技术，但众所周知，它容易受到对抗输入的影响。我们研究黑匣子对抗设置，其中$R #{k X n}中的固定、隐藏的草图矩阵A将R ##中的多维载体v $\映射到$R #中的低维草图A v #中的低维草图A v，并且对手可以查询系统以获得从草图计算的大约ell 2-模估计。   我们提出了一种通用的、非适应性攻击，它使用波浪形（O）（$k ' 2 $）查询，要么导致规范估计失败，要么构建一个对抗性输入，而查询分布的最佳估计器（由攻击使用）失败。该攻击对草图矩阵和估计器完全不可知：它适用于任何线性草图和任何查询响应器，包括随机化、自适应或针对查询分布定制的那些。   我们的下限结构与波浪形（Omega）的已知上限（$k ' 2 $）紧密匹配，该上限由Johnson Lindenstrauss变换和AMS草图的专门估计器实现。除了草图之外，我们的结果还揭示了图像分类中与对抗攻击的结构相似之处，凸显了压缩表示的基本漏洞。



## **13. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

像网络钓鱼者一样说话：基于LLM的语音网络钓鱼分类器攻击 cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.

摘要: 语音网络钓鱼（网络钓鱼）仍然是网络安全领域的一个持续威胁，它通过有说服力的言论来利用人类的信任。虽然基于机器学习（ML）的分类器在检测恶意通话记录方面表现出了希望，但它们仍然容易受到保留语义内容的对抗性操纵的影响。在这项研究中，我们探索了一种新型的攻击载体，其中利用大型语言模型（LLM）来生成对抗性的钓鱼笔录，以逃避检测，同时保持欺骗意图。我们构建了一个系统性攻击管道，该管道采用即时工程和语义混淆来使用四个商业LLM来转换现实世界的视频脚本。针对在现实世界的韩国钓鱼数据集（KorCCCViD）上训练的多个ML分类器进行评估生成的成绩单，并进行统计测试。我们的实验表明，LLM生成的成绩单对于基于ML的分类器在实践和统计上都有效。特别是，GPT-4 o制作的文字记录显着降低了分类器的准确性（高达30.96%），同时保持了高的语义相似性（由BERTScore衡量）。此外，这些攻击兼具时间效率和成本效益，平均生成时间不到9秒，每次查询的财务成本可以忽略不计。结果强调了对更具弹性的钓鱼检测框架的迫切需要，并强调了LLM提供者必须实施更强有力的保障措施，防止在对抗性社会工程环境中迅速滥用。



## **14. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

使用具有指定概率操纵的白盒对抗攻击对DNN模型进行所有权验证 cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2505.17579v2) [paper-pdf](http://arxiv.org/pdf/2505.17579v2)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.

摘要: 在本文中，我们提出了一种新的框架，用于图像分类任务的深度神经网络（DNN）模型的所有权验证。它允许合法所有者和第三方验证型号身份，而无需出示原始型号。我们假设一个灰盒场景，其中未经授权的用户拥有从原始模型非法复制的模型，在云环境中提供服务，用户抛出图像并接收分类结果作为输出类的概率分布。该框架应用白盒对抗攻击来将特定类的输出概率与指定值对齐。由于对原始模型的了解，它使所有者能够生成此类对抗性示例。我们通过引入控制参数，提出了一种基于迭代快速梯度符号法（FGSM）的简单但有效的对抗攻击方法。实验结果证实了使用对抗攻击识别DNN模型的有效性。



## **15. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

拉开帷幕：通过对比辅助网络的无监督对抗检测 cs.CV

Accepted at SafeMM-AI @ ICCV 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.09110v2) [paper-pdf](http://arxiv.org/pdf/2502.09110v2)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.

摘要: 深度学习模型广泛应用于安全关键应用中，但仍然容易受到对抗攻击--难以察觉的扰动，会显着降低模型性能。传统的防御机制主要专注于增强模型稳健性或独立检测对抗输入。在这项工作中，我们提出了一种通过对比辅助网络（U-CAN）的无监督对抗检测，以发现辅助特征表示中的对抗行为，而不需要对抗示例。U-CAN嵌入在目标模型的选定中间层中。这些辅助网络由投影层和基于ArcFace的线性层组成，可以细化特征表示，以更有效地区分良性输入和对抗输入。跨多个数据集（CIFAR-10、Mammals和ImageNet的一个子集）和架构（ResNet-50、VGG-16和ViT）的综合实验表明，我们的方法超越了现有的无监督对抗检测技术，在针对四种不同的攻击方法的情况下获得了优异的F1分数。提出的框架为增强深度学习系统的安全性和可靠性提供了可扩展且有效的解决方案。



## **16. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

优质文本，稳健的视觉：语言在增强视觉语言模型视觉稳健性方面的作用 cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.

摘要: 保护预先训练的视觉语言模型（VLM）（例如CLIP）免受对抗攻击至关重要，因为这些模型广泛用于各种零射击任务，包括图像分类。然而，现有的用于鲁棒微调的对抗训练（AT）方法在很大程度上忽视了语言在增强视觉鲁棒性方面的作用。具体来说，（1）监督AT方法依赖于短文本（例如，类标签）来生成对抗性扰动，导致对训练数据中的对象类的过度匹配，以及（2）无监督AT避免了这种过度匹配，但由于缺乏语义指导，对于实际的文本引导对抗性攻击仍然次优。为了解决这些限制，我们提出了优质文本引导的对抗微调（QT-AFT），它利用训练期间的高质量字幕来引导对抗示例远离图像中存在的各种语义。这使得视觉编码器即使在对抗性噪音下也能够稳健地识别更广泛的图像特征，从而增强各种下游任务的稳健性。QT-AFT克服了现有方法的关键弱点--监督AT中的过度匹配和无监督AT中缺乏语义意识--实现了最先进的零射击对抗鲁棒性和清晰的准确性，并在16个零射击数据集中进行了评估。此外，我们的全面研究揭示了语言在增强视觉鲁棒性方面作用的几个关键见解;例如，除了对象名称之外，描述对象属性进一步增强了零镜头鲁棒性。我们的研究结果指出了未来工作的紧迫方向--以稳健的视觉表示学习为中心的高质量语言监督。



## **17. Pulse-Level Simulation of Crosstalk Attacks on Superconducting Quantum Hardware**

超导体量子硬件串话攻击的脉冲级模拟 quant-ph

This paper has been accepted to the Security, Privacy, and Resilience  Workshop at IEEE Quantum Week (QCE 2025) and will appear in the workshop  proceedings

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16181v1) [paper-pdf](http://arxiv.org/pdf/2507.16181v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers poses a severe security threat, allowing adversaries to induce targeted errors across tenant boundaries by injecting carefully engineered pulses. We present a simulation-based study of active crosstalk attacks at the pulse level, analyzing how adversarial control of pulse timing, shape, amplitude, and coupling can disrupt a victim's computation. Our framework models the time-dependent dynamics of a three-qubit system in the rotating frame, capturing both always-on couplings and injected drive pulses. We examine two attack strategies: attacker-first (pulse before victim operation) and victim-first (pulse after), and systematically identify the pulse and coupling configurations that cause the largest logical errors. Protocol-level experiments on quantum coin flip and XOR classification circuits show that some protocols are highly vulnerable to these attacks, while others remain robust. Based on these findings, we discuss practical methods for detection and mitigation to improve security in quantum cloud platforms.

摘要: 多租户高温量子计算机中的硬件串烧构成了严重的安全威胁，使对手能够通过注入精心设计的脉冲来跨越租户边界引发有针对性的错误。我们对脉冲级的主动串话攻击进行了一项基于模拟的研究，分析脉冲定时、形状、幅度和耦合的对抗控制如何扰乱受害者的计算。我们的框架对旋转框架中三量子位系统的时间相关动态进行建模，捕获始终在线的耦合和注入的驱动脉冲。我们研究了两种攻击策略：攻击者优先（受害者操作前脉冲）和受害者优先（受害者操作后脉冲），并系统性地识别导致最大逻辑错误的脉冲和耦合配置。在量子硬币翻转和XOR分类电路上进行的协议级实验表明，一些协议非常容易受到这些攻击，而另一些协议仍然很健壮。基于这些发现，我们讨论了检测和缓解的实用方法，以提高量子云平台的安全性。



## **18. Attacking interpretable NLP systems**

攻击可解释的NLP系统 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16164v1) [paper-pdf](http://arxiv.org/pdf/2507.16164v1)

**Authors**: Eldor Abdukhamidov, Tamer Abuhmed, Joanna C. S. Santos, Mohammed Abuhamad

**Abstract**: Studies have shown that machine learning systems are vulnerable to adversarial examples in theory and practice. Where previous attacks have focused mainly on visual models that exploit the difference between human and machine perception, text-based models have also fallen victim to these attacks. However, these attacks often fail to maintain the semantic meaning of the text and similarity. This paper introduces AdvChar, a black-box attack on Interpretable Natural Language Processing Systems, designed to mislead the classifier while keeping the interpretation similar to benign inputs, thus exploiting trust in system transparency. AdvChar achieves this by making less noticeable modifications to text input, forcing the deep learning classifier to make incorrect predictions and preserve the original interpretation. We use an interpretation-focused scoring approach to determine the most critical tokens that, when changed, can cause the classifier to misclassify the input. We apply simple character-level modifications to measure the importance of tokens, minimizing the difference between the original and new text while generating adversarial interpretations similar to benign ones. We thoroughly evaluated AdvChar by testing it against seven NLP models and three interpretation models using benchmark datasets for the classification task. Our experiments show that AdvChar can significantly reduce the prediction accuracy of current deep learning models by altering just two characters on average in input samples.

摘要: 研究表明，机器学习系统在理论和实践中容易受到对抗性例子的影响。之前的攻击主要集中在利用人类和机器感知之间差异的视觉模型上，而基于文本的模型也成为这些攻击的受害者。然而，这些攻击往往无法保持文本的语义含义和相似性。本文介绍了AdvChar，这是一种针对可解释自然语言处理系统的黑匣子攻击，旨在误导分类器，同时保持解释类似于良性输入，从而利用对系统透明度的信任。AdvChar通过对文本输入进行不太明显的修改来实现这一目标，迫使深度学习分类器做出错误的预测并保留原始解释。我们使用以解释为中心的评分方法来确定最关键的标记，这些标记在更改时可能会导致分类器对输入进行错误分类。我们应用简单的字符级修改来衡量代币的重要性，最大限度地减少原始文本和新文本之间的差异，同时生成类似于良性解释的对抗性解释。我们通过使用分类任务的基准数据集针对七个NLP模型和三个解释模型进行测试，彻底评估了AdvChar。我们的实验表明，AdvChar只需平均改变输入样本中的两个字符即可显着降低当前深度学习模型的预测准确性。



## **19. DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT**

DP 2Guard：一种用于工业物联网的轻量级、拜占庭鲁棒的隐私保护联邦学习计划 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16134v1) [paper-pdf](http://arxiv.org/pdf/2507.16134v1)

**Authors**: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

**Abstract**: Privacy-Preserving Federated Learning (PPFL) has emerged as a secure distributed Machine Learning (ML) paradigm that aggregates locally trained gradients without exposing raw data. To defend against model poisoning threats, several robustness-enhanced PPFL schemes have been proposed by integrating anomaly detection. Nevertheless, they still face two major challenges: (1) the reliance on heavyweight encryption techniques results in substantial communication and computation overhead; and (2) single-strategy defense mechanisms often fail to provide sufficient robustness against adaptive adversaries. To overcome these challenges, we propose DP2Guard, a lightweight PPFL framework that enhances both privacy and robustness. DP2Guard leverages a lightweight gradient masking mechanism to replace costly cryptographic operations while ensuring the privacy of local gradients. A hybrid defense strategy is proposed, which extracts gradient features using singular value decomposition and cosine similarity, and applies a clustering algorithm to effectively identify malicious gradients. Additionally, DP2Guard adopts a trust score-based adaptive aggregation scheme that adjusts client weights according to historical behavior, while blockchain records aggregated results and trust scores to ensure tamper-proof and auditable training. Extensive experiments conducted on two public datasets demonstrate that DP2Guard effectively defends against four advanced poisoning attacks while ensuring privacy with reduced communication and computation costs.

摘要: 隐私保护联邦学习（PPFL）已成为一种安全的分布式机器学习（ML）范式，可以在不暴露原始数据的情况下聚合本地训练的梯度。为了抵御模型中毒威胁，通过集成异常检测提出了几种鲁棒性增强的PPFL方案。尽管如此，它们仍然面临两个主要挑战：（1）对重量级加密技术的依赖导致大量通信和计算负担;（2）单策略防御机制通常无法提供足够的鲁棒性来对抗自适应对手。为了克服这些挑战，我们提出了DP 2Guard，这是一个轻量级PPFL框架，可以增强隐私性和稳健性。DP 2Guard利用轻量级的梯度屏蔽机制来取代昂贵的加密操作，同时确保本地梯度的隐私。提出了一种混合防御策略，利用奇异值分解和cos相似度提取梯度特征，并应用集群算法有效识别恶意梯度。此外，DP 2Guard采用基于信任分数的自适应聚合方案，根据历史行为调整客户权重，而区块链记录聚合结果和信任分数，以确保防篡改和可审计的训练。在两个公共数据集上进行的大量实验表明，DP 2Guard可以有效防御四种高级中毒攻击，同时确保隐私并降低通信和计算成本。



## **20. DP-TLDM: Differentially Private Tabular Latent Diffusion Model**

DP-TLDM：差异私人表格潜在扩散模型 cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2403.07842v2) [paper-pdf](http://arxiv.org/pdf/2403.07842v2)

**Authors**: Chaoyi Zhu, Jiayi Tang, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. Till date, the prior focus on limited types of tabular synthesizers and a small number of privacy attacks, particularly on Generative Adversarial Networks, and overlooks membership inference attacks and defense strategies, i.e., differential privacy. Motivated by the conundrum of keeping high data quality and low privacy risk of synthetic data tables, we propose DPTLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DPTLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DPTLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.

摘要: 来自生成模型的合成数据成为保护隐私的数据共享解决方案。此类合成数据集应与原始数据相似，而不会透露可识别的私人信息。迄今为止，之前的重点是有限类型的表格合成器和少数隐私攻击，特别是生成式对抗网络，并忽视了成员资格推断攻击和防御策略，即差异隐私。出于保持合成数据表的高数据质量和低隐私风险的难题的动机，我们提出了DPTLDM，即差异私有表格潜在扩散模型，它由一个用于编码表格数据的自动编码器网络和一个用于合成潜在表的潜在扩散模型组成。遵循新兴的f-DP框架，我们应用DP-BCD结合批量剪裁来训练自动编码器，并使用分离值作为隐私指标，以更好地捕捉DP算法的隐私收益。我们的实证评估表明，DPTLDM能够实现有意义的理论隐私保证，同时还显着提高合成数据的实用性。具体来说，与其他DP保护的表格生成模型相比，DPTLDM将合成质量在数据相似性方面平均提高了35%，下游任务的实用性提高了15%，数据区分性提高了50%，同时保留了相当水平的隐私风险。



## **21. Erasing Conceptual Knowledge from Language Models**

从语言模型中删除概念知识 cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2410.02760v3) [paper-pdf](http://arxiv.org/pdf/2410.02760v3)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we introduce Erasure of Language Memory (ELM), a principled approach to concept-level unlearning that operates by matching distributions defined by the model's own introspective classification capabilities. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the language model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative evaluation reveals that ELM-modified models achieve near-random performance on assessments targeting erased concepts, while simultaneously preserving generation coherence, maintaining benchmark performance on unrelated tasks, and exhibiting strong robustness to adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info

摘要: 在这项工作中，我们引入了语言记忆擦除（ELM），这是一种概念级去学习的原则方法，通过匹配模型自己的内省分类能力定义的分布来操作。我们的主要见解是，有效的取消学习应该利用模型评估其自身知识的能力，使用语言模型本身作为分类器来识别和降低生成与不需要的概念相关的内容的可能性。ELM应用此框架来创建有针对性的低等级更新，以降低特定概念内容的生成概率，同时保留模型更广泛的功能。我们展示了ELM在生物安全、网络安全和文学领域擦除任务方面的功效。比较评估表明，ELM修改后的模型在针对已删除概念的评估中实现了近乎随机的性能，同时保持世代一致性，在不相关任务上保持基准性能，并对对抗性攻击表现出强大的鲁棒性。我们的代码、数据和训练模型可在https://elm.baulab.info上获取



## **22. Disrupting Semantic and Abstract Features for Better Adversarial Transferability**

破坏语义和抽象特征以提高对抗性可转移性 cs.CV

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.16052v1) [paper-pdf](http://arxiv.org/pdf/2507.16052v1)

**Authors**: Yuyang Luo, Xiaosen Wang, Zhijin Ge, Yingzhe He

**Abstract**: Adversarial examples pose significant threats to deep neural networks (DNNs), and their property of transferability in the black-box setting has led to the emergence of transfer-based attacks, making it feasible to target real-world applications employing DNNs. Among them, feature-level attacks, where intermediate features are perturbed based on feature importance weight matrix computed from transformed images, have gained popularity. In this work, we find that existing feature-level attacks primarily manipulate the semantic information to derive the weight matrix. Inspired by several works that find CNNs tend to focus more on high-frequency components (a.k.a. abstract features, e.g., texture, edge, etc.), we validate that transforming images in the high-frequency space also improves transferability. Based on this finding, we propose a balanced approach called Semantic and Abstract FEatures disRuption (SAFER). Specifically, SAFER conducts BLOCKMIX on the input image and SELF-MIX on the frequency spectrum when computing the weight matrix to highlight crucial features. By using such a weight matrix, we can direct the attacker to disrupt both semantic and abstract features, leading to improved transferability. Extensive experiments on the ImageNet dataset also demonstrate the effectiveness of our method in boosting adversarial transferability.

摘要: 对抗性示例对深度神经网络（DNN）构成了重大威胁，其在黑匣子环境中的可移植性导致了基于传输的攻击的出现，使得使用DNN针对现实世界的应用程序成为可能。其中，特征级攻击越来越流行，其中根据从变换图像计算出的特征重要性权重矩阵来扰乱中间特征。在这项工作中，我们发现现有的特征级攻击主要操纵语义信息来推导权重矩阵。受几项作品的启发，这些作品发现CNN倾向于更多地关注高频分量（又名抽象特征，例如，纹理、边缘等），我们验证了在高频空间中变换图像也可以提高可移植性。基于这一发现，我们提出了一种平衡的方法，称为语义和抽象特征分解（SAGER）。具体来说，SAGER在计算权重矩阵以突出关键特征时对输入图像进行BLOCKMIX，并对频谱进行SELF-MIX。通过使用这样的权重矩阵，我们可以指示攻击者破坏语义和抽象特征，从而提高可移植性。ImageNet数据集上的大量实验也证明了我们的方法在提高对抗可移植性方面的有效性。



## **23. Does More Inference-Time Compute Really Help Robustness?**

更多的推理时间计算真的有助于鲁棒性吗？ cs.AI

Preprint

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15974v1) [paper-pdf](http://arxiv.org/pdf/2507.15974v1)

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal

**Abstract**: Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications.

摘要: 最近，Zaremba等人证明，增加推理时计算可以提高大型专有推理LLM的鲁棒性。在本文中，我们首先展示了较小规模的开源模型（例如，DeepSeek R1、Qwen 3、Phi-reason）还可以受益于使用简单的预算强制策略的推理时扩展。更重要的是，我们揭示并批判性地检查了之前工作中的一个隐含假设：中间推理步骤对对手是隐藏的。通过放松这一假设，我们确定了一个重要的安全风险，直观的动机和经验验证的逆尺度律：如果中间的推理步骤变得显式访问，增加推理时间计算一致降低模型的鲁棒性。最后，我们讨论了具有隐藏推理链的模型仍然容易受到攻击的实际场景，例如具有工具集成推理和高级推理提取攻击的模型。我们的研究结果共同表明，推理时间缩放的鲁棒性优势在很大程度上取决于对抗性设置和部署上下文。我们敦促从业者在安全敏感的实际应用中应用推理时间缩放之前仔细权衡这些微妙的权衡。



## **24. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges**

沼泽上的对冲基金：分析区块链桥梁中的模式、漏洞和防御措施 cs.ET

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.06156v3) [paper-pdf](http://arxiv.org/pdf/2507.06156v3)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.

摘要: 区块链桥梁已成为实现不同区块链网络互操作性的重要基础设施，每月桥梁交易量超过240亿美元。然而，随着它们的日益普及，安全漏洞也不成比例地增加，使它们成为Web 3中最大的财务损失来源。为了实现跨链生态系统的稳健和可持续发展，了解和解决这些脆弱性至关重要。在这项研究中，我们对区块链桥梁设计和安全进行了全面的系统化。我们定义了三个桥梁安全先验，正式确定了13个突出桥梁的架构结构，并确定了23个基于现实世界区块链漏洞的攻击向量。在此基础上，我们评估了43种有代表性的攻击场景，并引入了一个分层的威胁模型，该模型可以捕获源链、链下和目标链组件的安全故障。   我们在静态代码和交易网络层面的分析揭示了反复出现的设计缺陷，特别是在访问控制、验证者信任假设和验证逻辑方面，并根据交易级跟踪识别了对抗行为的关键模式。为了支持未来的发展，我们提出了一个决策框架的桥梁架构设计，以及防御机制，如分层验证和断路器。这项工作为评估桥梁安全性提供了数据驱动的基础，并为标准化弹性跨链基础设施奠定了基础。



## **25. Sparsification Under Siege: Defending Against Poisoning Attacks in Communication-Efficient Federated Learning**

围攻下的稀疏化：在通信高效的联邦学习中防御中毒攻击 cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2505.01454v4) [paper-pdf](http://arxiv.org/pdf/2505.01454v4)

**Authors**: Zhiyong Jin, Runhua Xu, Chao Li, Yizhong Liu, Jianxin Li, James Joshi

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it faces significant challenges in communication efficiency and vulnerability to poisoning attacks. While sparsification techniques mitigate communication overhead by transmitting only critical model parameters, they inadvertently amplify security risks: adversarial clients can exploit sparse updates to evade detection and degrade model performance. Existing defense mechanisms, designed for standard FL communication scenarios, are ineffective in addressing these vulnerabilities within sparsified FL. To bridge this gap, we propose FLARE, a novel federated learning framework that integrates sparse index mask inspection and model update sign similarity analysis to detect and mitigate poisoning attacks in sparsified FL. Extensive experiments across multiple datasets and adversarial scenarios demonstrate that FLARE significantly outperforms existing defense strategies, effectively securing sparsified FL against poisoning attacks while maintaining communication efficiency.

摘要: 联合学习（FL）支持跨分布式客户端的协作模型训练，同时保护数据隐私，但它在通信效率和中毒攻击的脆弱性方面面临着重大挑战。虽然稀疏化技术通过仅传输关键模型参数来减轻通信负担，但它们无意中放大了安全风险：对抗性客户端可以利用稀疏更新来逃避检测并降低模型性能。为标准FL通信场景设计的现有防御机制无法解决稀疏FL中的这些漏洞。为了弥合这一差距，我们提出了LGA，一个新颖的联邦学习框架，集成了稀疏索引屏蔽检查和模型更新符号相似性分析，以检测和减轻稀疏FL中的中毒攻击。跨多个数据集和对抗场景的广泛实验表明，LGA显着优于现有的防御策略，有效地保护稀疏FL免受中毒攻击，同时保持通信效率。



## **26. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

对企业LLM系统的多阶段即时推理攻击 cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.

摘要: 部署在企业环境中的大型语言模型（LLM）（例如，作为Microsoft 365 Copilot）面临着新颖的安全挑战。一个关键威胁是提示推理攻击：对手将看似良性的提示链接在一起，以逐渐提取机密数据。本文对企业LLM上下文中的多阶段提示推理攻击进行了全面研究。我们模拟了现实的攻击场景，其中攻击者使用温和的查询和间接提示注入来利用集成了私人公司数据的LLM。我们为这些多回合推理攻击开发了一个正式的威胁模型，并使用概率论、优化框架和信息论泄漏界限对其进行分析。这些攻击被证明可以可靠地从LLM的上下文中泄露敏感信息（例如，内部SharePoint文档或电子邮件），即使已采取标准安全措施。   我们提出并评估防御措施来对抗此类攻击，包括统计异常检测、细粒度访问控制、即时清理技术以及对LLM部署的架构修改。每个防御都有数学分析或实验模拟的支持。例如，我们在基于隐私的差异训练下推导出信息泄露的界限，并演示了一种异常检测方法，该方法可以标记具有高AUR的多回合攻击。我们还引入了一种名为“聚光灯”的方法，该方法使用输入转换来隔离不受信任的提示内容，从而将攻击成功率降低一个数量级。最后，我们为联合深度防御策略提供了正式的概念证明和经验验证。我们的工作强调，在企业环境中保护LLM需要超越单轮即时过滤，转向针对攻击和防御的整体、多阶段视角。



## **27. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

统一XAI的无导扩散总管约束梯度 cs.CV

CVPR 2025 (poster), 19 pages, 5 figures

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2411.15265v2) [paper-pdf](http://arxiv.org/pdf/2411.15265v2)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.

摘要: 基于对象的方法是一个典型的可解释性技术家族，尤其是对于基于图像的模型。尽管如此，它们也有几个缺点，因为它们（1）需要白盒访问模型，（2）容易受到对抗性攻击，（3）产生脱离图像多管的属性，导致解释实际上不忠实于模型，并且不符合人类感知。为了克服这些挑战，我们引入了无求导扩散Manifold约束子（FreeMCG），这是一种新颖的方法，与传统梯度相比，它可以作为给定神经网络可解释性的改进基础。具体来说，通过利用集合卡尔曼过滤器和扩散模型，我们推导出投影到数据集上的模型梯度的无导逼近，仅需要访问模型的输出。我们通过将FreeMCG应用于反事实生成和特征归因来证明它的有效性，这两个传统上被视为不同的任务。通过对这两项任务、反事实解释和特征归因的全面评估，我们表明我们的方法可以产生最先进的结果，同时保留了XAI工具所期望的基本属性。



## **28. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

坏与好的传输攻击：解释和增强多模式大型语言模型之间的对抗性传输 cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.

摘要: 多模式大型语言模型（MLLM）在跨模式交互中表现出出色的性能，但它们也存在对抗性漏洞。特别是，对抗性例子的可移植性仍然是一个持续的挑战。本文具体分析了MLLM之间对抗性转移性的表现，并确定了影响这一特征的关键因素。我们发现，MLLM的可移植性存在于具有相同视觉编码器的跨LLM场景中，并指出可能影响可移植性的\underline{\textit{两个关键因素}}。我们提供了两种语义级数据增强方法：添加图像补丁（AIP）和印刷增强可移植性方法（TATM），它们增强了对抗性示例跨MLLM的可移植性。为了探索对现实世界的潜在影响，我们利用了两项可能产生负面和积极社会影响的任务：\ding{182}有害内容插入和\ding{183}信息保护。



## **29. Scaling Decentralized Learning with FLock**

使用Flock扩展分散式学习 cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **30. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

猫混淆推理LLM：推理模型的查询不可知对抗触发器 cs.CL

Accepted to CoLM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2503.01781v2) [paper-pdf](http://arxiv.org/pdf/2503.01781v2)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.

摘要: 我们通过引入查询不可知的对抗触发器（简短、不相关的文本，当附加到数学问题时，会系统性地误导模型输出错误答案，而不改变问题的语义，来研究为逐步解决问题而训练的推理模型的稳健性。我们提出CatAttack，这是一种自动迭代攻击管道，用于在较弱、较便宜的代理模型（DeepSeek V3）上生成触发器，并成功将它们转移到DeepSeek R1和DeepSeek R1-蒸馏-Qwen-32 B等更高级的推理目标模型，导致目标模型生成错误答案的可能性增加300%以上。例如，在任何数学问题上添加“有趣的事实：猫一生中大部分时间都在睡觉”都会导致模型出错的机会增加一倍多。我们的研究结果凸显了推理模型中的关键漏洞，揭示了即使是最先进的模型仍然容易受到微妙的对抗输入的影响，从而引发了安全性和可靠性的担忧。CatAttack触发具有模型响应的数据集可在https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers上获取。



## **31. Defective Convolutional Networks**

有缺陷的卷积网络 cs.CV

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/1911.08432v3) [paper-pdf](http://arxiv.org/pdf/1911.08432v3)

**Authors**: Tiange Luo, Tianle Cai, Mengxiao Zhang, Siyu Chen, Di He, Liwei Wang

**Abstract**: Robustness of convolutional neural networks (CNNs) has gained in importance on account of adversarial examples, i.e., inputs added as well-designed perturbations that are imperceptible to humans but can cause the model to predict incorrectly. Recent research suggests that the noises in adversarial examples break the textural structure, which eventually leads to wrong predictions. To mitigate the threat of such adversarial attacks, we propose defective convolutional networks that make predictions relying less on textural information but more on shape information by properly integrating defective convolutional layers into standard CNNs. The defective convolutional layers contain defective neurons whose activations are set to be a constant function. As defective neurons contain no information and are far different from standard neurons in its spatial neighborhood, the textural features cannot be accurately extracted, and so the model has to seek other features for classification, such as the shape. We show extensive evidence to justify our proposal and demonstrate that defective CNNs can defense against black-box attacks better than standard CNNs. In particular, they achieve state-of-the-art performance against transfer-based attacks without any adversarial training being applied.

摘要: 由于对抗性示例，卷积神经网络（CNN）的鲁棒性变得越来越重要，即输入作为精心设计的扰动添加，人类无法察觉，但可能导致模型预测错误。最近的研究表明，对抗性例子中的噪音打破了文本结构，最终导致错误的预测。为了减轻此类对抗攻击的威胁，我们提出了有缺陷的卷积网络，通过将有缺陷的卷积层正确集成到标准CNN中，这些网络减少了对纹理信息的依赖，而更多地依赖形状信息来进行预测。有缺陷的卷积层包含有缺陷的神经元，其激活被设置为恒定函数。由于有缺陷的神经元不包含信息，并且在其空间附近与标准神经元相去甚远，因此无法准确提取纹理特征，因此模型必须寻找其他特征进行分类，例如形状。我们展示了大量的证据来证明我们的提议是合理的，并证明有缺陷的CNN比标准CNN更能防御黑匣子攻击。特别是，它们在不应用任何对抗训练的情况下就能获得针对基于传输的攻击的最先进性能。



## **32. ROBAD: Robust Adversary-aware Local-Global Attended Bad Actor Detection Sequential Model**

ROBAD：稳健的对手感知本地-全球参与不良行为者检测序列模型 cs.LG

15 pages, 12 tables

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15067v1) [paper-pdf](http://arxiv.org/pdf/2507.15067v1)

**Authors**: Bing He, Mustaque Ahamad, Srijan Kumar

**Abstract**: Detecting bad actors is critical to ensure the safety and integrity of internet platforms. Several deep learning-based models have been developed to identify such users. These models should not only accurately detect bad actors, but also be robust against adversarial attacks that aim to evade detection. However, past deep learning-based detection models do not meet the robustness requirement because they are sensitive to even minor changes in the input sequence. To address this issue, we focus on (1) improving the model understanding capability and (2) enhancing the model knowledge such that the model can recognize potential input modifications when making predictions. To achieve these goals, we create a novel transformer-based classification model, called ROBAD (RObust adversary-aware local-global attended Bad Actor Detection model), which uses the sequence of user posts to generate user embedding to detect bad actors. Particularly, ROBAD first leverages the transformer encoder block to encode each post bidirectionally, thus building a post embedding to capture the local information at the post level. Next, it adopts the transformer decoder block to model the sequential pattern in the post embeddings by using the attention mechanism, which generates the sequence embedding to obtain the global information at the sequence level. Finally, to enrich the knowledge of the model, embeddings of modified sequences by mimicked attackers are fed into a contrastive-learning-enhanced classification layer for sequence prediction. In essence, by capturing the local and global information (i.e., the post and sequence information) and leveraging the mimicked behaviors of bad actors in training, ROBAD can be robust to adversarial attacks. Extensive experiments on Yelp and Wikipedia datasets show that ROBAD can effectively detect bad actors when under state-of-the-art adversarial attacks.

摘要: 检测不良行为者对于确保互联网平台的安全性和完整性至关重要。已经开发了几个基于深度学习的模型来识别此类用户。这些模型不仅应该准确地检测不良行为者，而且还应该对旨在逃避检测的对抗性攻击具有鲁棒性。然而，过去的基于深度学习的检测模型不满足稳健性要求，因为它们对输入序列中的哪怕是微小的变化也很敏感。为了解决这个问题，我们重点关注（1）提高模型理解能力和（2）增强模型知识，以便模型在做出预测时能够识别潜在的输入修改。为了实现这些目标，我们创建了一个新颖的基于转换器的分类模型，称为ROBAD（ROBust对手感知本地-全球参与的坏演员检测模型），它使用用户帖子序列来生成用户嵌入来检测坏演员。特别是，ROBAD首先利用Transformer编码器块对每个帖子进行双向编码，从而构建帖子嵌入以捕获帖子级别的本地信息。接下来，采用Transformer解码器块，利用注意机制对后嵌入中的序列模式进行建模，生成序列嵌入，以获取序列级的全局信息。最后，为了丰富模型的知识，模拟攻击者对修改后的序列的嵌入被输入对比学习增强分类层以进行序列预测。本质上，通过捕获本地和全球信息（即，发布和序列信息）并利用训练中不良行为者的模仿行为，ROBAD可以对对抗性攻击具有鲁棒性。对Yelp和维基百科数据集的大量实验表明，ROBAD可以在最先进的对抗攻击下有效地检测不良行为者。



## **33. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection**

DeRAG：通过提示注入对多个检索增强生成应用程序的黑匣子对抗攻击 cs.AI

Accepted by KDD Workshop on Prompt Optimization 2025

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15042v1) [paper-pdf](http://arxiv.org/pdf/2507.15042v1)

**Authors**: Jerry Wang, Fang Yu

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy.

摘要: 对抗提示攻击可以通过重新排序检索增强生成（RAG）系统以产生错误的输出来显着改变它们的可靠性。在本文中，我们提出了一种应用差异进化（DE）来优化基于RAG的问答的对抗性提示后缀的新方法。我们的方法是无梯度的，将RAG管道视为黑匣子，并进化出一系列候选后缀，以最大化目标错误文档的检索排名，使其更接近现实世界场景。我们对BEIR QA数据集进行了实验，以评估多个检索应用程序下特定检索等级阈值下的攻击成功率。我们的结果表明，与GDPP对密集检索器和PRADA对稀疏检索器相比，基于DE的即时优化获得了有竞争力的（在某些情况下更高）成功率，同时在对抗性后缀中仅使用少量令牌（<=5个令牌）。此外，我们引入了一种可读写性感知的后缀构建策略，该策略通过韦尔奇t检验MLM负对log似然性的统计学显着降低来验证。通过使用基于BERT的对抗性后缀检测器进行评估，我们表明DE生成的后缀可以逃避检测，从而产生近乎偶然的检测准确性。



## **34. Adversarial Destabilization Attacks to Direct Data-Driven Control**

直接数据驱动控制的对抗性去稳定攻击 eess.SY

15 pages

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14863v1) [paper-pdf](http://arxiv.org/pdf/2507.14863v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control methods, specifically for the linear quadratic regulator problem, to adversarial perturbations in collected data used for controller synthesis. We consider stealthy attacks that subtly manipulate offline-collected data to destabilize the resulting closed-loop system while evading detection. To generate such perturbations, we propose the Directed Gradient Sign Method (DGSM) and its iterative variant (I-DGSM), adaptations of the fast gradient sign method originally developed for neural networks, which align perturbations with the gradient of the spectral radius of the closed-loop matrix to reduce stability. A key contribution is an efficient gradient computation technique based on implicit differentiation through the Karush-Kuhn-Tucker conditions of the underlying semidefinite program, enabling scalable and exact gradient evaluation without repeated optimization computations. To defend against these attacks, we propose two defense strategies: a regularization-based approach that enhances robustness by suppressing controller sensitivity to data perturbations and a robust data-driven control approach that guarantees closed-loop stability within bounded perturbation sets. Extensive numerical experiments on benchmark systems show that adversarial perturbations with magnitudes up to ten times smaller than random noise can destabilize controllers trained on corrupted data and that the proposed defense strategies effectively mitigate attack success rates while maintaining control performance. Additionally, we evaluate attack transferability under partial knowledge scenarios, highlighting the practical importance of protecting training data confidentiality.

摘要: 本研究调查了直接数据驱动控制方法（特别是线性二次调节器问题）对用于控制器综合的收集数据中对抗性扰动的脆弱性。我们考虑的是隐蔽攻击，这些攻击巧妙地操纵离线收集的数据，以破坏由此产生的闭环系统的稳定性，同时逃避检测。为了产生此类扰动，我们提出了有向梯度符号法（DGSM）及其迭代变体（I-DGSM），这是最初为神经网络开发的快速梯度符号法的改编，它将扰动与闭环矩阵的谱半径的梯度对齐以降低稳定性。一个关键贡献是一种基于通过底层半定程序的Karush-Kuhn-Tucker条件进行隐式求导的高效梯度计算技术，无需重复优化计算即可实现可扩展且准确的梯度评估。为了抵御这些攻击，我们提出了两种防御策略：一种基于正规化的方法，通过抑制控制器对数据扰动的敏感性来增强鲁棒性，另一种鲁棒的数据驱动控制方法，保证有界扰动集中的闭环稳定性。对基准系统进行的大量数值实验表明，幅度比随机噪音小十倍的对抗性扰动可能会破坏根据损坏数据训练的控制器的稳定性，并且提出的防御策略有效地降低了攻击成功率，同时保持控制性能。此外，我们还评估了部分知识场景下的攻击可转移性，强调了保护训练数据机密性的实际重要性。



## **35. Data-Plane Telemetry to Mitigate Long-Distance BGP Hijacks**

数据平面遥感缓解长距离BNP劫持 cs.NI

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14842v1) [paper-pdf](http://arxiv.org/pdf/2507.14842v1)

**Authors**: Satadal Sengupta, Hyojoon Kim, Daniel Jubas, Maria Apostolaki, Jennifer Rexford

**Abstract**: Poor security of Internet routing enables adversaries to divert user data through unintended infrastructures (hijack). Of particular concern -- and the focus of this paper -- are cases where attackers reroute domestic traffic through foreign countries, exposing it to surveillance, bypassing legal privacy protections, and posing national security threats. Efforts to detect and mitigate such attacks have focused primarily on the control plane while data-plane signals remain largely overlooked. In particular, change in propagation delay caused by rerouting offers a promising signal: the change is unavoidable and the increased propagation delay is directly observable from the affected networks. In this paper, we explore the practicality of using delay variations for hijack detection, addressing two key questions: (1) What coverage can this provide, given its heavy dependence on the geolocations of the sender, receiver, and adversary? and (2) Can an always-on latency-based detection system be deployed without disrupting normal network operations? We observe that for 86% of victim-attacker country pairs in the world, mid-attack delays exceed pre-attack delays by at least 25% in real deployments, making delay-based hijack detection promising. To demonstrate practicality, we design HiDe, which reliably detects delay surges from long-distance hijacks at line rate. We measure HiDe's accuracy and false-positive rate on real-world data and validate it with ethically conducted hijacks.

摘要: 互联网路由的安全性较差使对手能够通过意外的基础设施（劫持）转移用户数据。特别令人担忧的情况--也是本文的重点--是攻击者通过外国重新路由国内流量，使其受到监视，绕过法律隐私保护，并构成国家安全威胁。检测和减轻此类攻击的努力主要集中在控制平面上，而数据平面信号在很大程度上仍然被忽视。特别是，重新路由引起的传播延迟变化提供了一个有希望的信号：这种变化是不可避免的，并且传播延迟的增加可以从受影响的网络中直接观察到。在本文中，我们探讨了使用延迟变化的劫持检测的实用性，解决两个关键问题：（1）这可以提供什么样的覆盖范围，鉴于其严重依赖于发送者，接收器和对手的地理位置？以及（2）可以在不中断正常网络操作的情况下部署始终在线的基于延迟的检测系统吗？我们观察到，对于世界上86%的受害者-攻击者国家对，在实际部署中，攻击中期延迟超过攻击前延迟至少25%，这使得基于延迟的劫持检测很有希望。为了证明实用性，我们设计了HiDe，它可以以线速可靠地检测来自长距离劫持的延迟浪涌。我们在现实世界数据上衡量HiDe的准确性和假阳性率，并通过道德劫持来验证它。



## **36. Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree**

利用HTML可访问性树实现LLM Web代理的间接提示注入攻击 cs.CR

EMNLP 2025 System Demonstrations Submission

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14799v1) [paper-pdf](http://arxiv.org/pdf/2507.14799v1)

**Authors**: Sam Johnson, Viet Pham, Thai Le

**Abstract**: This work demonstrates that LLM-based web navigation agents offer powerful automation capabilities but are vulnerable to Indirect Prompt Injection (IPI) attacks. We show that adversaries can embed universal adversarial triggers in webpage HTML to hijack agent behavior that utilizes the accessibility tree to parse HTML, causing unintended or malicious actions. Using the Greedy Coordinate Gradient (GCG) algorithm and a Browser Gym agent powered by Llama-3.1, our system demonstrates high success rates across real websites in both targeted and general attacks, including login credential exfiltration and forced ad clicks. Our empirical results highlight critical security risks and the need for stronger defenses as LLM-driven autonomous web agents become more widely adopted. The system software (https://github.com/sej2020/manipulating-web-agents) is released under the MIT License, with an accompanying publicly available demo website (http://lethaiq.github.io/attack-web-llm-agent).

摘要: 这项工作表明，基于LLM的网络导航代理提供了强大的自动化功能，但很容易受到间接提示注入（IPI）攻击。我们表明，对手可以在网页HTML中嵌入通用对抗触发器，以劫持利用可访问性树解析HTML的代理行为，从而导致意外或恶意操作。我们的系统使用贪婪坐标梯度（GCG）算法和由Llama-3.1支持的浏览器健身房代理，在针对性和一般攻击（包括登录凭证泄露和强制广告点击）中，在真实网站上都表现出很高的成功率。我们的实证结果凸显了随着LLM驱动的自主网络代理被更广泛采用，关键的安全风险以及对更强防御的需求。系统软件（https：//github.com/sej2020/manipulating-web-agents）在MIT许可证下发布，附带一个公开的演示网站（http：//lethaiq.github.io/attack-web-llm-agent）。



## **37. GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks**

GCC-Spam：通过GAN，对比学习和字符相似性网络进行垃圾邮件检测 cs.LG

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14679v1) [paper-pdf](http://arxiv.org/pdf/2507.14679v1)

**Authors**: Zixin Xu, Zhijie Wang, Zhiyuan Pan

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.

摘要: 互联网上垃圾文本呈指数级增长，需要强大的检测机制来减轻信息泄露和社会不稳定等风险。这项工作解决了两个主要挑战：垃圾邮件发送者采用的对抗策略和标记数据的稀缺性。我们提出了一种新颖的垃圾邮件文本检测框架GCC-Spam，该框架集成了三项核心创新。首先，字符相似性网络捕获正音特征以对抗字符混淆攻击，并进一步生成用于下游分类的句子嵌入。其次，对比学习通过优化垃圾邮件和正常文本之间的潜在空间距离来增强区分性。第三，生成对抗网络（GAN）生成真实的伪垃圾邮件样本，以缓解数据稀缺性，同时提高模型稳健性和分类准确性。对现实世界数据集的广泛实验表明，我们的模型优于基线方法，以显着更少的标记示例实现了更高的检测率。



## **38. VTarbel: Targeted Label Attack with Minimal Knowledge on Detector-enhanced Vertical Federated Learning**

VTarbel：对检测器增强的垂直联邦学习了解最少的有针对性的标签攻击 cs.CR

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14625v1) [paper-pdf](http://arxiv.org/pdf/2507.14625v1)

**Authors**: Juntao Tan, Anran Li, Quanchao Liu, Peng Ran, Lan Zhang

**Abstract**: Vertical federated learning (VFL) enables multiple parties with disjoint features to collaboratively train models without sharing raw data. While privacy vulnerabilities of VFL are extensively-studied, its security threats-particularly targeted label attacks-remain underexplored. In such attacks, a passive party perturbs inputs at inference to force misclassification into adversary-chosen labels. Existing methods rely on unrealistic assumptions (e.g., accessing VFL-model's outputs) and ignore anomaly detectors deployed in real-world systems. To bridge this gap, we introduce VTarbel, a two-stage, minimal-knowledge attack framework explicitly designed to evade detector-enhanced VFL inference. During the preparation stage, the attacker selects a minimal set of high-expressiveness samples (via maximum mean discrepancy), submits them through VFL protocol to collect predicted labels, and uses these pseudo-labels to train estimated detector and surrogate model on local features. In attack stage, these models guide gradient-based perturbations of remaining samples, crafting adversarial instances that induce targeted misclassifications and evade detection. We implement VTarbel and evaluate it against four model architectures, seven multimodal datasets, and two anomaly detectors. Across all settings, VTarbel outperforms four state-of-the-art baselines, evades detection, and retains effective against three representative privacy-preserving defenses. These results reveal critical security blind spots in current VFL deployments and underscore urgent need for robust, attack-aware defenses.

摘要: 垂直联邦学习（VFL）使具有不相交特征的多方能够协作训练模型，而无需共享原始数据。虽然VFL的隐私漏洞被广泛研究，其安全威胁，特别是有针对性的标签攻击，仍然未充分探索。在这种攻击中，被动方在推理时干扰输入，以迫使错误分类为对手选择的标签。现有的方法依赖于不切实际的假设（例如，访问VFL模型的输出）并忽略现实世界系统中部署的异常检测器。为了弥合这一差距，我们引入了VTarbel，这是一个两阶段、最低限度知识攻击框架，专门设计用于规避检测器增强的VFL推断。在准备阶段，攻击者选择一组最小的高表达性样本（通过最大平均差异），通过VFL协议提交它们以收集预测标签，并使用这些伪标签来训练估计的检测器和本地特征的代理模型。在攻击阶段，这些模型引导剩余样本的基于梯度的扰动，精心设计引发有针对性的错误分类并逃避检测的对抗实例。我们实现VTarbel并针对四个模型架构、七个多模式数据集和两个异常检测器对其进行评估。在所有设置中，VTarbel的性能优于四个最先进的基线，逃避检测，并对三种代表性的隐私保护防御保持有效。这些结果揭示了当前VFL部署中的关键安全盲点，并强调了对强大的攻击感知防御的迫切需求。



## **39. 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving**

3DGAA：针对自动驾驶的真实且稳健的3D基于高斯的对抗攻击 cs.CV

Submitted to WACV 2026

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.09993v2) [paper-pdf](http://arxiv.org/pdf/2507.09993v2)

**Authors**: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

**Abstract**: Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. Existing 2D and 3D physical attacks, due to their focus on texture optimization, often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture optimization, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module that filters outliers to preserve geometric fidelity, and a physical augmentation module that simulates complex physical scenarios to enhance attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21\% to 7.38\%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks.

摘要: 基于摄像头的物体检测系统在自动驾驶中发挥着至关重要的作用，但它们在现实世界环境中仍然容易受到对抗威胁的影响。现有的2D和3D物理攻击由于关注纹理优化，通常难以平衡物理真实感和攻击鲁棒性。在这项工作中，我们提出了基于3D高斯的对抗性攻击（3DGAA），这是一种新型的对抗性对象生成框架，它利用3D高斯飞溅（3DGS）的完整14维参数化来以物理可实现的方式联合优化几何形状和外观。与依赖补丁或纹理优化的先前作品不同，3DGAA联合扰动几何属性（形状、比例、旋转）和外观属性（颜色、不透明度），以产生物理上真实且可转移的对抗对象。我们进一步引入了一个物理过滤模块，用于过滤异常值以保持几何保真度，以及一个物理增强模块，用于模拟复杂的物理场景以增强现实世界条件下的攻击概括性。我们使用微型车辆模型在虚拟基准和物理世界设置上评估3DGAA。实验结果表明，3DGAA将检测mAP从87.21%降低到7.38%，显着优于现有的3D物理攻击。此外，我们的方法在不同的物理条件下保持了高度的可移植性，展示了物理可实现的对抗攻击的新技术水平。



## **40. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

通过潜在对抗训练防御不可预见的失败模式 cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2403.05030v5) [paper-pdf](http://arxiv.org/pdf/2403.05030v5)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，人工智能系统有时会表现出有害的非预期行为。找到和修复这些问题具有挑战性，因为攻击面如此之大--无法彻底搜索可能引发有害行为的输入。红色团队和对抗训练（AT）通常用于提高稳健性，然而，从经验上看，它们很难修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需利用有关漏洞的知识或使用引发漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。在这里，我们使用它来防御没有引发失败模式的例子的失败模式。具体来说，我们使用LAT来删除后门并抵御持续的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常会提高对新型攻击的鲁棒性和干净数据的性能。这表明LAT可以成为一种有前途的工具，用于防御开发人员未明确识别的故障模式。



## **41. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

FedStrategist：一个用于联邦学习中自适应和鲁棒聚合的元学习框架 cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14322v1) [paper-pdf](http://arxiv.org/pdf/2507.14322v1)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.

摘要: 联邦学习（FL）为保护隐私的协作人工智能提供了一个范式，但其去中心化性质给建模中毒攻击带来了显着的漏洞。虽然存在许多静态防御，但它们的有效性高度依赖于上下文，通常无法对抗自适应对手或在异类数据环境中。本文介绍了FedStrategist，这是一种新型的元学习框架，它将稳健聚合重新定义为实时、成本感知的控制问题。我们设计了一个轻量级的上下文强盗代理，它基于实时诊断指标从防御库中动态选择最佳聚合规则。通过全面的实验，我们证明没有单一的静态规则是普遍最优的。我们表明，我们的适应性代理能够在不同的场景中成功学习更好的策略，包括“克鲁姆有利”的环境以及针对旨在中和特定诊断信号的复杂“隐形”对手。至关重要的是，我们分析了自相矛盾的场景，即非稳健基线实现了高但受到损害的准确性，并证明我们的代理学习了保守的策略来优先考虑模型完整性。此外，我们证明了代理的策略是可以通过单个“风险容忍度”参数来控制的，允许从业者显式地管理性能和安全性之间的权衡。我们的工作提供了一种新的、实用的、可分析的方法来创建弹性和智能的去中心化人工智能系统。



## **42. An Adversarial-Driven Experimental Study on Deep Learning for RF Fingerprinting**

基于对抗驱动的射频指纹深度学习实验研究 cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14109v1) [paper-pdf](http://arxiv.org/pdf/2507.14109v1)

**Authors**: Xinyu Cao, Bimal Adhikari, Shangqing Zhao, Jingxian Wu, Yanjun Pan

**Abstract**: Radio frequency (RF) fingerprinting, which extracts unique hardware imperfections of radio devices, has emerged as a promising physical-layer device identification mechanism in zero trust architectures and beyond 5G networks. In particular, deep learning (DL) methods have demonstrated state-of-the-art performance in this domain. However, existing approaches have primarily focused on enhancing system robustness against temporal and spatial variations in wireless environments, while the security vulnerabilities of these DL-based approaches have often been overlooked. In this work, we systematically investigate the security risks of DL-based RF fingerprinting systems through an adversarial-driven experimental analysis. We observe a consistent misclassification behavior for DL models under domain shifts, where a device is frequently misclassified as another specific one. Our analysis based on extensive real-world experiments demonstrates that this behavior can be exploited as an effective backdoor to enable external attackers to intrude into the system. Furthermore, we show that training DL models on raw received signals causes the models to entangle RF fingerprints with environmental and signal-pattern features, creating additional attack vectors that cannot be mitigated solely through post-processing security methods such as confidence thresholds.

摘要: 射频（RF）指纹识别提取无线电设备独特的硬件缺陷，已成为零信任架构和5G网络以外的一种有前途的物理层设备识别机制。特别是，深度学习（DL）方法在该领域展示了最先进的性能。然而，现有方法主要集中在增强系统针对无线环境中的时间和空间变化的鲁棒性，而这些基于DL的方法的安全漏洞常常被忽视。在这项工作中，我们通过对抗驱动的实验分析系统地调查了基于DL的RF指纹识别系统的安全风险。我们观察到一个一致的误分类行为的DL模型下域的变化，其中一个设备经常被误分类为另一个特定的。我们的分析基于广泛的现实世界的实验表明，这种行为可以被利用作为一个有效的后门，使外部攻击者入侵系统。此外，我们还证明了在原始接收信号上训练DL模型会导致模型将RF指纹与环境和信号模式特征纠缠在一起，从而产生额外的攻击向量，这些攻击向量不能仅仅通过后处理安全方法（如置信度阈值）来缓解。



## **43. CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

CDUpatch：针对双模式可见红外探测器的颜色驱动通用对抗补丁攻击 cs.CV

Accepted by ACMMM 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2504.10888v2) [paper-pdf](http://arxiv.org/pdf/2504.10888v2)

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios.

摘要: 对抗补丁被广泛用于评估现实世界场景中物体检测系统的稳健性。这些补丁最初设计用于欺骗单模式检测器（例如，可见光或红外），并且最近已扩展到瞄准可见-红外双模式检测器。然而，现有的双模式对抗补丁攻击在不同物理场景中的攻击有效性有限。为了解决这个问题，我们提出了CDUpatch，这是一种针对跨尺度、视图和场景可见红外物体检测器的通用跨模式补丁攻击。具体来说，我们观察到颜色变化会导致不同水平的热吸收，从而导致红外成像中的温度差异。利用这一属性，我们提出了一种RGB到红外适配器，可以将RGB补丁映射到红外补丁，从而实现跨模式补丁的统一优化。通过学习对抗性斑块上的最佳颜色分布，我们可以操纵其热响应并生成对抗性红外纹理。此外，我们引入了多尺度剪裁策略并构建了一个新的可见红外数据集MSDS rone，其中包含不同尺度和视角的飞行器图像。这些数据增强策略增强了我们补丁在现实世界条件下的稳健性。对四个基准数据集进行实验（例如，DroneVehicles、LLVIP、VisDrone）表明我们的方法优于数字领域中现有的补丁攻击。广泛的物理测试进一步证实了跨规模、视图和场景的强大可移植性。



## **44. Bridging Local and Global Knowledge via Transformer in Board Games**

通过棋盘游戏中的Transformer连接本地和全球知识 cs.LG

Accepted by the Thirty-Fourth International Joint Conferences on  Artificial Intelligence (IJCAI-25)

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2410.05347v2) [paper-pdf](http://arxiv.org/pdf/2410.05347v2)

**Authors**: Yan-Ru Ju, Tai-Lin Wu, Chung-Chin Shih, Ti-Rong Wu

**Abstract**: Although AlphaZero has achieved superhuman performance in board games, recent studies reveal its limitations in handling scenarios requiring a comprehensive understanding of the entire board, such as recognizing long-sequence patterns in Go. To address this challenge, we propose ResTNet, a network that interleaves residual and Transformer blocks to bridge local and global knowledge. ResTNet improves playing strength across multiple board games, increasing win rate from 54.6% to 60.8% in 9x9 Go, 53.6% to 60.9% in 19x19 Go, and 50.4% to 58.0% in 19x19 Hex. In addition, ResTNet effectively processes global information and tackles two long-sequence patterns in 19x19 Go, including circular pattern and ladder pattern. It reduces the mean square error for circular pattern recognition from 2.58 to 1.07 and lowers the attack probability against an adversary program from 70.44% to 23.91%. ResTNet also improves ladder pattern recognition accuracy from 59.15% to 80.01%. By visualizing attention maps, we demonstrate that ResTNet captures critical game concepts in both Go and Hex, offering insights into AlphaZero's decision-making process. Overall, ResTNet shows a promising approach to integrating local and global knowledge, paving the way for more effective AlphaZero-based algorithms in board games. Our code is available at https://rlg.iis.sinica.edu.tw/papers/restnet.

摘要: 尽管AlphaZero在棋盘游戏中取得了超人的表现，但最近的研究揭示了它在处理需要全面了解整个棋盘的场景方面的局限性，例如识别Go中的长序列模式。为了应对这一挑战，我们提出了ResTNet，这是一个交织剩余块和Transformer块的网络，以连接本地和全球知识。ResTNet提高了多个棋盘游戏的玩法，将9x9 Go的胜率从54.6%提高到60.8%，19x19 Go的胜率从53.6%提高到60.9%，19x19 Hex的胜率从50.4%提高到58.0%。此外，ResTNet有效处理全球信息并解决19 x19 Go中的两种长序列模式，包括循环模式和阶梯模式。它将圆形模式识别的均方误差从2.58降低到1.07，并将针对对手程序的攻击概率从70.44%降低到23.91%。ResTNet还将阶梯模式识别准确率从59.15%提高到80.01%。通过可视化注意力地图，我们证明ResTNet捕捉了Go和Hex中的关键游戏概念，为AlphaZero的决策过程提供了见解。总体而言，ResTNet展示了一种整合本地和全球知识的有希望的方法，为棋盘游戏中更有效的基于AlphaZero的算法铺平了道路。我们的代码可在https://rlg.iis.sinica.edu.tw/papers/restnet上获取。



## **45. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

从言语到碰撞：法学硕士指导的评估和安全关键驾驶场景的对抗生成 cs.AI

Final Version and Paper Accepted at IEEE ITSC 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.02145v4) [paper-pdf](http://arxiv.org/pdf/2502.02145v4)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.

摘要: 确保自动驾驶汽车的安全需要基于虚拟环境的测试，这取决于安全关键场景的稳健评估和生成。到目前为止，研究人员已经使用基于情景的测试框架，这些框架严重依赖手工制作的场景作为安全指标。为了减少人类解释的工作量并克服这些方法的有限可扩展性，我们将大型语言模型（LLM）与结构化场景解析相结合，并提示工程技术自动评估和生成对安全至关重要的驾驶场景。我们引入了用于场景评估的Cartesian和以自我为中心的提示策略，以及一个对抗生成模块，该模块修改风险诱导车辆（自我攻击者）的轨迹以创建关键场景。我们使用2D仿真框架和多个预先训练的LLM来验证我们的方法。结果表明，该评估模块能够有效地检测碰撞场景，并推断出场景安全性.与此同时，新一代模块识别高风险代理并综合现实的安全关键场景。我们的结论是，LLM配备域知情的提示技术可以有效地评估和生成安全关键的驾驶场景，减少依赖手工制作的指标。我们在https://github.com/TUM-AVS/From-Words-to-Collisions上发布我们的开源代码和场景。



## **46. Adversarial Training Improves Generalization Under Distribution Shifts in Bioacoustics**

对抗性训练提高生物声学分布变化下的概括性 cs.LG

Work in progress

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13727v1) [paper-pdf](http://arxiv.org/pdf/2507.13727v1)

**Authors**: René Heinrich, Lukas Rauch, Bernhard Sick, Christoph Scholz

**Abstract**: Adversarial training is a promising strategy for enhancing model robustness against adversarial attacks. However, its impact on generalization under substantial data distribution shifts in audio classification remains largely unexplored. To address this gap, this work investigates how different adversarial training strategies improve generalization performance and adversarial robustness in audio classification. The study focuses on two model architectures: a conventional convolutional neural network (ConvNeXt) and an inherently interpretable prototype-based model (AudioProtoPNet). The approach is evaluated using a challenging bird sound classification benchmark. This benchmark is characterized by pronounced distribution shifts between training and test data due to varying environmental conditions and recording methods, a common real-world challenge. The investigation explores two adversarial training strategies: one based on output-space attacks that maximize the classification loss function, and another based on embedding-space attacks designed to maximize embedding dissimilarity. These attack types are also used for robustness evaluation. Additionally, for AudioProtoPNet, the study assesses the stability of its learned prototypes under targeted embedding-space attacks. Results show that adversarial training, particularly using output-space attacks, improves clean test data performance by an average of 10.5% relative and simultaneously strengthens the adversarial robustness of the models. These findings, although derived from the bird sound domain, suggest that adversarial training holds potential to enhance robustness against both strong distribution shifts and adversarial attacks in challenging audio classification settings.

摘要: 对抗训练是一种有前途的策略，可以增强模型针对对抗攻击的稳健性。然而，在音频分类的数据分布大幅变化下，它对概括性的影响在很大程度上仍未被探索。为了解决这一差距，这项工作研究了不同的对抗训练策略如何提高音频分类中的概括性能和对抗鲁棒性。该研究重点关注两种模型架构：传统的卷积神经网络（ConvNeXt）和本质上可解释的基于原型的模型（AudioProtoPNet）。使用具有挑战性的鸟声分类基准的方法进行评估。该基准测试的特点是，由于不同的环境条件和记录方法，训练和测试数据之间存在明显的分布变化，这是现实世界中常见的挑战。该调查探讨了两种对抗性训练策略：一种基于输出空间攻击，最大化分类损失函数，另一种基于嵌入空间攻击，旨在最大化嵌入相异性。这些攻击类型也用于鲁棒性评估。此外，对于AudioProtoPNet，该研究评估了其学习原型在有针对性的嵌入空间攻击下的稳定性。结果表明，对抗性训练，特别是使用输出空间攻击，将干净测试数据性能平均提高了10.5%，同时增强了模型的对抗性鲁棒性。这些发现虽然源自鸟类声音领域，但表明对抗训练有潜力增强针对具有挑战性的音频分类环境中强分布变化和对抗攻击的鲁棒性。



## **47. Breaking the Illusion of Security via Interpretation: Interpretable Vision Transformer Systems under Attack**

通过解释打破安全幻觉：可解释的视觉Transformer系统受到攻击 cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.14248v1) [paper-pdf](http://arxiv.org/pdf/2507.14248v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Hyoungshick Kim, Tamer Abuhmed

**Abstract**: Vision transformer (ViT) models, when coupled with interpretation models, are regarded as secure and challenging to deceive, making them well-suited for security-critical domains such as medical applications, autonomous vehicles, drones, and robotics. However, successful attacks on these systems can lead to severe consequences. Recent research on threats targeting ViT models primarily focuses on generating the smallest adversarial perturbations that can deceive the models with high confidence, without considering their impact on model interpretations. Nevertheless, the use of interpretation models can effectively assist in detecting adversarial examples. This study investigates the vulnerability of transformer models to adversarial attacks, even when combined with interpretation models. We propose an attack called "AdViT" that generates adversarial examples capable of misleading both a given transformer model and its coupled interpretation model. Through extensive experiments on various transformer models and two transformer-based interpreters, we demonstrate that AdViT achieves a 100% attack success rate in both white-box and black-box scenarios. In white-box scenarios, it reaches up to 98% misclassification confidence, while in black-box scenarios, it reaches up to 76% misclassification confidence. Remarkably, AdViT consistently generates accurate interpretations in both scenarios, making the adversarial examples more difficult to detect.

摘要: 视觉Transformer（ViT）模型与解释模型结合使用时，被认为是安全的且难以欺骗的，因此非常适合医疗应用、自动驾驶汽车、无人机和机器人等安全关键领域。然而，对这些系统的成功攻击可能会导致严重的后果。最近关于针对ViT模型的威胁的研究主要集中在生成最小的对抗性扰动，这些扰动可以以高置信度欺骗模型，而不考虑它们对模型解释的影响。尽管如此，使用解释模型可以有效地帮助检测对抗性例子。这项研究调查了Transformer模型对抗攻击的脆弱性，即使与解释模型相结合。我们提出了一种名为“AdViT”的攻击，它会生成能够误导给定Transformer模型及其耦合解释模型的对抗性示例。通过对各种Transformer模型和两个基于转换器的解释器的广泛实验，我们证明AdViT在白盒和黑盒场景中都实现了100%的攻击成功率。在白盒场景中，它达到高达98%的误分类置信度，而在黑盒场景中，它达到高达76%的误分类置信度。值得注意的是，AdViT在两种情况下都能一致地产生准确的解释，使对抗性示例更难检测。



## **48. GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention**

Gift：用户感知通过安全的概念保留，免疫扩散模型对抗恶意微调 cs.CR

Warning: This paper contains NSFW content. Reader discretion is  advised

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13598v1) [paper-pdf](http://arxiv.org/pdf/2507.13598v1)

**Authors**: Amro Abdalla, Ismail Shaheen, Dan DeGenaro, Rupayan Mallick, Bogdan Raita, Sarah Adel Bargal

**Abstract**: We present GIFT: a {G}radient-aware {I}mmunization technique to defend diffusion models against malicious {F}ine-{T}uning while preserving their ability to generate safe content. Existing safety mechanisms like safety checkers are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model's ability to represent harmful concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe generative quality. Experimental results show that our method significantly impairs the model's ability to re-learn harmful concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks.

摘要: 我们赠送的礼物：一种{G}辐射感知的{I}信息技术，可保护扩散模型免受恶意{F}ine-{T}联合的影响，同时保留其生成安全内容的能力。安全检查器等现有的安全机制很容易被绕过，概念擦除方法在对抗性微调下也会失败。Gift通过将免疫定义为一个双层优化问题来解决这个问题：上层目标降低了模型使用表示噪音和最大化来表示有害概念的能力，而下层目标则保留了安全数据的性能。Gift在保持安全的生成质量的同时实现了对恶意微调的强大抵抗。实验结果表明，我们的方法显着削弱了模型在保持安全内容性能的同时重新学习有害概念的能力，为创建本质上更安全的、抵御对抗性微调攻击的生成模型提供了一个有希望的方向。



## **49. BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems**

AMPS：针对基于协作多智能体深度强化学习的系统的隐形后门杠杆攻击 cs.AI

12. arXiv admin note: substantial text overlap with arXiv:2409.07775

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2501.01593v2) [paper-pdf](http://arxiv.org/pdf/2501.01593v2)

**Authors**: Jing Fang, Saihao Yan, Xueyu Yin, Yinbo Yu, Chunwei Tian, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform malicious actions leading to failures or malicious goals. However, existing backdoor attacks suffer from several issues, e.g., instant trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor leverage attack against c-MADRL, BLAST, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the period to perform malicious actions. This method can guarantee the stealthiness and practicality of BLAST. Secondly, we hack the original reward function of the backdoor agent via unilateral guidance to inject BLAST, so as to achieve the \textit{leverage attack effect} that can pry open the entire multi-agent system via a single backdoor agent. We evaluate our BLAST against 3 classic c-MADRL algorithms (VDN, QMIX, and MAPPO) in 2 popular c-MADRL environments (SMAC and Pursuit), and 2 existing defense mechanisms. The experimental results demonstrate that BLAST can achieve a high attack success rate while maintaining a low clean performance variance rate.

摘要: 最近的研究表明，合作多智能体深度强化学习（c-MADRL）面临后门攻击的威胁。一旦观察到后门触发，它将执行导致失败或恶意目标的恶意操作。然而，现有的后门攻击存在几个问题，例如，即时触发模式缺乏隐蔽性，后门由额外的网络训练或激活，或者所有代理都是后门的。为此，在本文中，我们提出了一种针对c-MADRL的新型后门杠杆攻击，即BST，该攻击通过仅将后门嵌入到单个代理中来攻击整个多代理团队。首先，我们引入对手时空行为模式作为后门触发器，而不是手动注入的固定视觉模式或即时状态，并控制执行恶意行为的时间。该方法可以保证BST的隐蔽性和实用性。其次，我们通过单边引导破解后门代理原有的奖励功能，注入BST，从而达到通过单个后门代理撬开整个多代理系统的\textit{杠杆攻击效应}。我们在2种流行的c-MADRL环境（SMAC和Pursuit）和2种现有防御机制中针对3种经典c-MADRL算法（VDN、QMIX和MAPPO）评估了我们的BST。实验结果表明，AMPS可以在保持较低的干净性能变异率的同时实现高攻击成功率。



## **50. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

STACK：对LLM Safeguard Pipelines的对抗性攻击 cs.CL

Fixed typos (including Figure 1), amended GPU-hours rather than days,  clarified ReNeLLM prompt modifications

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2506.24068v2) [paper-pdf](http://arxiv.org/pdf/2506.24068v2)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.

摘要: 前沿人工智能开发人员依靠多层保障措施来防止人工智能系统的灾难性滥用。Anthropic使用这样的防御管道来保护他们最新的Claude 4 Opus模型，包括Google DeepMind和OpenAI在内的其他前沿开发商承诺很快部署类似的防御。然而，此类管道的安全性尚不清楚，之前评估或攻击这些管道的工作有限。我们通过开发和组建开源防御管道来解决这一差距。首先，我们发现一种新型的几次激发输入和输出分类器在三次攻击和两个数据集中优于最先进的开权保护模型ShieldGemma，将灾难性滥用数据集ClearHarm的攻击成功率（ASO）降低至0%。其次，我们引入了一个STaged AttaCK（STACK）过程，该过程在ClearHarm上实现了71%的ASB，针对少量镜头提示的分类器管道进行黑匣子攻击。最后，我们还在传输环境中评估了STACK，实现了33%的ASB，提供了初步证据，证明在不访问目标管道的情况下设计攻击是可行的。最后，我们建议开发人员可以用来阻止分阶段攻击的具体缓解措施。



