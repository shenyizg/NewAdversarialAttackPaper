# Latest Adversarial Attack Papers
**update at 2025-04-17 11:39:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Human Aligned Compression for Robust Models**

鲁棒模型的人类一致压缩 cs.CV

Presented at the Workshop AdvML at CVPR 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12255v1) [paper-pdf](http://arxiv.org/pdf/2504.12255v1)

**Authors**: Samuel Räber, Andreas Plesner, Till Aczel, Roger Wattenhofer

**Abstract**: Adversarial attacks on image models threaten system robustness by introducing imperceptible perturbations that cause incorrect predictions. We investigate human-aligned learned lossy compression as a defense mechanism, comparing two learned models (HiFiC and ELIC) against traditional JPEG across various quality levels. Our experiments on ImageNet subsets demonstrate that learned compression methods outperform JPEG, particularly for Vision Transformer architectures, by preserving semantically meaningful content while removing adversarial noise. Even in white-box settings where attackers can access the defense, these methods maintain substantial effectiveness. We also show that sequential compression--applying rounds of compression/decompression--significantly enhances defense efficacy while maintaining classification performance. Our findings reveal that human-aligned compression provides an effective, computationally efficient defense that protects the image features most relevant to human and machine understanding. It offers a practical approach to improving model robustness against adversarial threats.

摘要: 对图像模型的对抗性攻击通过引入导致错误预测的不可感知的扰动来威胁系统的鲁棒性。我们研究了作为防御机制的人类对齐习得有损压缩，将两种习得模型（HiFiC和ELIC）与不同质量级别的传统JPEG进行比较。我们对ImageNet子集的实验表明，学习的压缩方法通过保留语义有意义的内容同时去除对抗性噪音而优于JPEG，特别是对于Vision Transformer架构。即使在攻击者可以访问防御的白盒环境中，这些方法也能保持相当大的有效性。我们还表明，顺序压缩-应用轮的压缩/解压缩-显着提高防御效率，同时保持分类性能。我们的研究结果表明，人类对齐的压缩提供了一种有效的，计算效率高的防御，可以保护与人类和机器理解最相关的图像特征。它提供了一种实用的方法来提高模型对对抗性威胁的鲁棒性。



## **2. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

LLM护栏中的快速注射和越狱检测 cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六种主要保护系统（包括微软的Azure Promise Shield和Meta的Promise Guard）进行测试，我们表明这两种方法都可以用来逃避检测，同时保持对抗性效用，在某些情况下实现高达100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **3. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全一致和取消学习 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 当前对LLM对抗鲁棒性的研究重点是自然语言空间中的离散输入操纵，其可以直接转移到闭源模型。然而，这种方法忽视了开源模型的稳定发展。随着开源模型功能的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM量身定制的攻击在很大程度上仍然未被探索。我们解决了这一研究空白并提出了嵌入空间攻击，该攻击直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地规避模型对齐并触发有害行为。此外，我们在取消学习的背景下提出了一种新颖的威胁模型，并表明嵌入空间攻击可以从多个数据集和模型中未学习的LLM中提取据称已删除的信息。我们的研究结果强调将空间攻击嵌入到开源LLM中作为重要威胁模型。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **4. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

具有不确定节点特征和不确定图结构的图卷积网络的形式化验证 cs.LG

published at Transactions on Machine Learning Research (TMLR) 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2404.15065v2) [paper-pdf](http://arxiv.org/pdf/2404.15065v2)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.

摘要: 图神经网络因其处理以图结构化的数据的独特能力而在机器学习领域变得越来越受欢迎。它们还应用于固有地发生扰动的安全关键环境中。然而，这些扰动需要我们在将神经网络部署到安全关键环境中之前对其进行正式验证，因为神经网络容易受到对抗性攻击。虽然存在关于神经网络形式验证的研究，但还没有任何工作验证通用图卷积网络架构的稳健性，因为节点特征和多个消息传递步骤中的图结构存在不确定性。这项工作通过使用（矩阵）多项分区的可达性分析明确保留基础计算中所有元素的非凸依赖性来解决这一研究空白。我们在三个流行的基准数据集上展示了我们的方法。



## **5. Attribute Inference Attacks for Federated Regression Tasks**

针对联邦回归任务的属性推理攻击 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2411.12697v2) [paper-pdf](http://arxiv.org/pdf/2411.12697v2)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.

摘要: 联合学习（FL）使多个客户端（如手机和物联网设备）能够协作训练全局机器学习模型，同时保持数据本地化。然而，最近的研究表明，FL的训练阶段容易受到重建攻击，如属性推理攻击（AIA），其中对手利用交换的消息和辅助公共信息来发现目标客户端的敏感属性。虽然这些攻击已经在分类任务的背景下进行了广泛的研究，但它们对回归任务的影响在很大程度上仍未被探索。在本文中，我们通过提出专门为FL环境中的回归任务设计的新的基于模型的AIA来解决这一差距。我们的方法考虑了对手可以窃听交换的消息或直接干扰训练过程的场景。我们使用现实世界的数据集针对最先进的方法对我们提出的攻击进行基准测试。结果表明重建准确性显着提高，特别是在FL中常见的异类客户端数据集中。我们基于模型的AIAs的功效使它们成为经验量化联邦回归任务隐私泄露的更好候选者。



## **6. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

RL SA-PFL：隐私保护联邦学习中具有模型不一致性检测的鲁棒轻量级安全聚合 cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2502.08989v2) [paper-pdf](http://arxiv.org/pdf/2502.08989v2)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.

摘要: 联合学习（FL）允许用户通过仅共享本地模型来协作训练全球机器学习模型，而无需将其私人数据暴露给中央服务器。这种分布式学习在数据隐私至关重要的场景中特别有吸引力，并且引起了行业和学术界的高度关注。然而，研究揭示了FL中的隐私漏洞，对手可能会从共享模型参数中推断敏感信息。在本文中，我们提出了一种高效的基于掩蔽的安全聚合方案，利用轻量级加密基元来降低隐私风险。与现有方法相比，我们的方案提供了几个优点。首先，整个FL训练课程只需要一个设置阶段，从而显着减少了通信负担。其次，它通过消除用户对用户交互的需要、利用中间服务器层和轻量级密钥协商方法来最大限度地减少用户端的负担。第三，该计划对用户退出具有高度弹性，用户可以在任何FL轮中加入。第四，它可以检测和防御恶意服务器活动，包括最近发现的模型不一致性攻击。最后，我们的方案确保了半诚实和恶意设置中的安全性。我们提供安全分析来正式证明我们方法的稳健性。此外，我们还实现了我们计划的端到端原型。我们进行了全面的实验和比较，结果表明它在通信和计算负担、功能性和安全性方面优于现有解决方案。



## **7. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

通过动态最大化优化改进普遍对抗扰动的推广 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2503.12793v3) [paper-pdf](http://arxiv.org/pdf/2503.12793v3)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.

摘要: 深度神经网络（DNN）容易受到普遍对抗性扰动（UPC）的影响。这些扰动经过精心设计，旨在在所有样本类中普遍欺骗目标模型。与特定于实例的对抗示例（AE）不同，生成UPC更加复杂，因为它们必须在广泛的数据样本和模型中进行概括。我们的研究表明，现有的通用攻击方法使用DNN和静态模型参数快照来优化UAP，但并没有充分利用DNN的潜力来生成更有效的UAP。我们建议使用动态模型-数据对来生成UAP，而不是针对具有固定训练集的静态DNN模型来优化UAP。特别是，我们引入了动态最大值优化策略，旨在在各种最佳模型-数据对上优化UAP。我们将这种方法称为DM-UAP。DM-UAP利用迭代的最大-最小-最小优化框架来细化模型-数据对，并结合课程UAP学习算法来彻底检查模型参数和数据的组合空间。ImageNet数据集的全面实验表明，提出的DM-UAP显着增强了UAP的跨样本通用性和跨模型可移植性。DM-UAP仅使用500个样本即可生成UAP，优于最先进的方法，愚弄率平均增加12.108%。



## **8. SemDiff: Generating Natural Unrestricted Adversarial Examples via Semantic Attributes Optimization in Diffusion Models**

SemDiff：通过扩散模型中的语义属性优化生成自然无限制对抗示例 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11923v1) [paper-pdf](http://arxiv.org/pdf/2504.11923v1)

**Authors**: Zeyu Dai, Shengcai Liu, Rui He, Jiahao Wu, Ning Lu, Wenqi Fan, Qing Li, Ke Tang

**Abstract**: Unrestricted adversarial examples (UAEs), allow the attacker to create non-constrained adversarial examples without given clean samples, posing a severe threat to the safety of deep learning models. Recent works utilize diffusion models to generate UAEs. However, these UAEs often lack naturalness and imperceptibility due to simply optimizing in intermediate latent noises. In light of this, we propose SemDiff, a novel unrestricted adversarial attack that explores the semantic latent space of diffusion models for meaningful attributes, and devises a multi-attributes optimization approach to ensure attack success while maintaining the naturalness and imperceptibility of generated UAEs. We perform extensive experiments on four tasks on three high-resolution datasets, including CelebA-HQ, AFHQ and ImageNet. The results demonstrate that SemDiff outperforms state-of-the-art methods in terms of attack success rate and imperceptibility. The generated UAEs are natural and exhibit semantically meaningful changes, in accord with the attributes' weights. In addition, SemDiff is found capable of evading different defenses, which further validates its effectiveness and threatening.

摘要: 无限制的对抗示例（UAE）允许攻击者在没有给出干净样本的情况下创建无限制的对抗示例，对深度学习模型的安全性构成严重威胁。最近的作品利用扩散模型来生成UAE。然而，由于简单地在中间潜在噪音中进行优化，这些UAE往往缺乏自然性和不可感知性。有鉴于此，我们提出了SemDiff，这是一种新型的无限制对抗攻击，它探索有意义属性的扩散模型的语义潜在空间，并设计了一种多属性优化方法，以确保攻击成功，同时保持生成的UAE的自然性和不可感知性。我们在三个高分辨率数据集（包括CelebA-HQ、AFHQ和ImageNet）上对四项任务进行了广泛的实验。结果表明，SemDiff在攻击成功率和不可感知性方面优于最先进的方法。生成的UAE是自然的，并根据属性的权重表现出有意义的语义变化。此外，SemDiff被发现能够规避不同的防御，这进一步验证了其有效性和威胁性。



## **9. RAB$^2$-DEF: Dynamic and explainable defense against adversarial attacks in Federated Learning to fair poor clients**

RAB $' 2 $-DEF：针对联邦学习中的对抗攻击的动态且可解释的防御，以应对贫困客户 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.08244v2) [paper-pdf](http://arxiv.org/pdf/2410.08244v2)

**Authors**: Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: At the same time that artificial intelligence is becoming popular, concern and the need for regulation is growing, including among other requirements the data privacy. In this context, Federated Learning is proposed as a solution to data privacy concerns derived from different source data scenarios due to its distributed learning. The defense mechanisms proposed in literature are just focused on defending against adversarial attacks and the performance, leaving aside other important qualities such as explainability, fairness to poor quality clients, dynamism in terms of attacks configuration and generality in terms of being resilient against different kinds of attacks. In this work, we propose RAB$^2$-DEF, a $\textbf{r}$esilient $\textbf{a}$gainst $\textbf{b}\text{yzantine}$ and $\textbf{b}$ackdoor attacks which is $\textbf{d}$ynamic, $\textbf{e}$xplainable and $\textbf{f}$air to poor clients using local linear explanations. We test the performance of RAB$^2$-DEF in image datasets and both byzantine and backdoor attacks considering the state-of-the-art defenses and achieve that RAB$^2$-DEF is a proper defense at the same time that it boosts the other qualities towards trustworthy artificial intelligence.

摘要: 在人工智能日益流行的同时，人们对监管的担忧和需求也在增长，其中包括数据隐私等要求。在此背景下，联邦学习因其分布式学习而被提出作为解决不同源数据场景中产生的数据隐私问题的解决方案。文献中提出的防御机制只是专注于防御对抗性攻击和性能，而忽略了其他重要的品质，例如可解释性、对质量较差的客户端的公平性、攻击配置方面的动态性以及针对不同类型攻击的韧性的一般性。在这项工作中，我们提出了RAB $' 2 $-DEF、a $\textBF{r}$esilient $\textBF{a}$gð $\textBF{b}\text{yzantine}$和$\textBF{b}$ackdoor攻击，这是$\textBF{d}$Thomic、$\textBF{e}$xplanable和$\textBF{f}$air使用本地线性解释对贫困客户端进行的攻击。我们测试了RAB $^2 $-DEF在图像数据集和拜占庭攻击和后门攻击中的性能，并考虑到最先进的防御方法，得出RAB $^2 $-DEF是一种适当的防御方法，同时它提高了其他质量，使人工智能变得可信。



## **10. Support is All You Need for Certified VAE Training**

支持就是您认证VAE培训所需的一切 cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11831v1) [paper-pdf](http://arxiv.org/pdf/2504.11831v1)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.

摘要: 变分自动编码器（VAE）变得越来越受欢迎，并部署在安全关键应用中。在此类应用程序中，我们希望为对抗性攻击下的性能提供经过认证的概率保证。我们提出了一种新的方法CIVET，用于VAE的认证培训。CIVET取决于一个关键见解，即我们可以通过将误差限制在潜在层精心选择的支持集上来限制最坏情况的VAE误差。我们以数学方式展示了这一点，并利用这一见解提出了一种新颖的训练算法。我们在对不同数据集（无线和视觉应用领域）、架构和扰动幅度的广泛评估中表明，我们的方法优于SOTA方法，可以在强大的稳健性保证下实现良好的标准性能。



## **11. PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection**

PSBD：预测转变不确定性解锁后门检测 cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.05826v2) [paper-pdf](http://arxiv.org/pdf/2406.05826v2)

**Authors**: Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang

**Abstract**: Deep neural networks are susceptible to backdoor attacks, where adversaries manipulate model predictions by inserting malicious samples into the training data. Currently, there is still a significant challenge in identifying suspicious training data to unveil potential backdoor samples. In this paper, we propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. PSBD is motivated by an intriguing Prediction Shift (PS) phenomenon, where poisoned models' predictions on clean data often shift away from true labels towards certain other labels with dropout applied during inference, while backdoor samples exhibit less PS. We hypothesize PS results from the neuron bias effect, making neurons favor features of certain classes. PSBD identifies backdoor training samples by computing the Prediction Shift Uncertainty (PSU), the variance in probability values when dropout layers are toggled on and off during model inference. Extensive experiments have been conducted to verify the effectiveness and efficiency of PSBD, which achieves state-of-the-art results among mainstream detection methods. The code is available at https://github.com/WL-619/PSBD.

摘要: 深度神经网络容易受到后门攻击，攻击者通过将恶意样本插入训练数据来操纵模型预测。目前，在识别可疑的训练数据以揭示潜在的后门样本方面仍然存在重大挑战。在本文中，我们提出了一种新的方法，预测移位后门检测（PSBD），利用基于不确定性的方法，需要最少的未标记的干净的验证数据。PSBD的动机是一个有趣的预测偏移（PS）现象，其中中毒模型对干净数据的预测经常从真实标签转向某些其他标签，在推理过程中应用dropout，而后门样本表现出较少的PS。我们假设PS的结果是神经元偏见效应，使神经元更喜欢某些类别的特征。PSBD通过计算预测漂移不确定性（PSO）来识别后门训练样本，即模型推理期间打开和关闭丢失层时的概率值的方差。人们进行了大量的实验来验证PSBD的有效性和效率，它取得了主流检测方法中最先进的结果。该代码可在https://github.com/WL-619/PSBD上获取。



## **12. Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset**

迈向网络上安全合成图像生成：多模式稳健的NSFW防御和百万规模数据集 cs.CV

Short Paper The Web Conference

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11707v1) [paper-pdf](http://arxiv.org/pdf/2504.11707v1)

**Authors**: Muhammad Shahid Muneer, Simon S. Woo

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: https://github.com/shahidmuneer/multimodal-nsfw-defense.

摘要: 在过去的几年里，我们见证了文本到图像（T2 I）模型的显着成功及其在网络上的广泛使用。对使T2 I模型产生超真实图像的广泛研究引发了新的担忧，例如生成不安全工作（NSFW）网络内容和污染网络社会。为了帮助防止滥用T2 I模型并为用户创建更安全的网络环境，这些模型中使用了NSFW过滤器和事后安全检查等功能。然而，最近的工作揭示了这些方法如何很容易无法防止滥用。特别是，对文本和图像模式的对抗攻击很容易胜过防御措施。%利用这一点导致人们越来越担心防止对文本和图像模式的对抗性攻击。此外，目前还没有包括提示和图像对以及对抗性示例的稳健多模式NSFW数据集。这项工作提出了使用开源扩散模型生成的百万级提示和图像数据集。其次，我们开发了一种多模式防御来区分安全和NSFW文本和图像，该防御系统能够强大地抵御对抗攻击，并直接缓解当前的挑战。我们广泛的实验表明，我们的模型在准确性和召回率方面相对于现有的SOTA NSFW检测方法表现良好，大幅降低了多模式对抗性攻击场景中的攻击成功率（ASB）。代码：https://github.com/shahidmuneer/multimodal-nsfw-defense。



## **13. Learning to Learn Transferable Generative Attack for Person Re-Identification**

学习学习用于人员重新识别的可转移生成攻击 cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2409.04208v2) [paper-pdf](http://arxiv.org/pdf/2409.04208v2)

**Authors**: Yuan Bian, Min Liu, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning-based person re-identification (re-id) models are widely employed in surveillance systems and inevitably inherit the vulnerability of deep networks to adversarial attacks. Existing attacks merely consider cross-dataset and cross-model transferability, ignoring the cross-test capability to perturb models trained in different domains. To powerfully examine the robustness of real-world re-id models, the Meta Transferable Generative Attack (MTGA) method is proposed, which adopts meta-learning optimization to promote the generative attacker producing highly transferable adversarial examples by learning comprehensively simulated transfer-based cross-model\&dataset\&test black-box meta attack tasks. Specifically, cross-model\&dataset black-box attack tasks are first mimicked by selecting different re-id models and datasets for meta-train and meta-test attack processes. As different models may focus on different feature regions, the Perturbation Random Erasing module is further devised to prevent the attacker from learning to only corrupt model-specific features. To boost the attacker learning to possess cross-test transferability, the Normalization Mix strategy is introduced to imitate diverse feature embedding spaces by mixing multi-domain statistics of target models. Extensive experiments show the superiority of MTGA, especially in cross-model\&dataset and cross-model\&dataset\&test attacks, our MTGA outperforms the SOTA methods by 21.5\% and 11.3\% on mean mAP drop rate, respectively. The code of MTGA will be released after the paper is accepted.

摘要: 基于深度学习的人员重新识别（re-id）模型被广泛应用于监控系统中，并且不可避免地继承了深度网络对抗攻击的脆弱性。现有的攻击只考虑了跨数据集和跨模型的可移植性，忽略了交叉测试对不同领域训练的模型的干扰能力。为了有力地检查现实世界re-id模型的稳健性，提出了Meta可转移生成攻击（MTGA）方法，该方法采用元学习优化，通过全面学习模拟的基于转移的交叉模型\&数据集\&测试黑匣子Meta攻击任务，促进生成攻击者产生高度可转移的对抗性示例。具体来说，首先通过为元训练和元测试攻击过程选择不同的re-id模型和数据集来模拟跨模型和数据集的黑匣子攻击任务。由于不同的模型可能关注不同的特征区域，因此进一步设计了扰动随机擦除模块，以防止攻击者学习仅破坏特定于模型的特征。为了促进攻击者学习具有交叉测试可移植性，引入了规范化混合策略，通过混合目标模型的多域统计数据来模仿不同的特征嵌入空间。大量实验表明了MTGA的优越性，特别是在跨模型数据集和跨模型数据集测试攻击中，我们的MTGA在平均mAP下降率方面分别比SOTA方法高出21.5%和11.3%。论文被接受后，MTGA的代码将发布。



## **14. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

Chypnosis：使用基于欠压的静态侧通道攻击进行秘密提取 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11633v1) [paper-pdf](http://arxiv.org/pdf/2504.11633v1)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.

摘要: 越来越多的静态物理侧通道攻击允许对手通过探测电路的持续状态来提取秘密。激光逻辑状态成像（LLSI）、阻抗分析（IA）和静态功率分析等技术都属于这一类。这些攻击要求目标数据在特定的持续时间内保持不变，这通常需要停止电路的时钟。有些方法还依赖于调制芯片的电源电压来探测电路。然而，对时钟或电压的篡改通常被认为是可检测的，因为安全芯片通常部署传感器，在检测到此类异常时擦除敏感数据。此外，许多安全设备使用内部时钟源，使得外部时钟控制不可行。在这项工作中，我们引入了一类新型的静态侧通道攻击，称为Chypnosis，它使对手能够通过快速欠电压诱导休眠状态来冻结芯片的内部时钟，然后使用静态侧通道提取秘密。我们证明，通过将芯片的电压快速降低到标准名义水平以下，攻击者可以绕过时钟和电压传感器，并将芯片置于所谓的停电条件，其中芯片的晶体管停止切换，但易失性存储器（例如，人字拖和RAM）仍然保留其数据。我们通过将AMD VGA置于休眠状态来测试对它们的攻击。我们表明，不仅所有时钟源都被停用，而且各种时钟和电压传感器也无法检测到篡改事件。随后，我们展示了使用两种静态攻击（LLSI和IA）从休眠芯片中成功恢复秘密位的方法。最后，我们讨论了可以整合到未来设计中的潜在对策。



## **15. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）在整个过程中提供语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。本文的研究可以进一步引起人们对多媒体信息安全的关注。



## **16. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

显而易见的不可见威胁：LLM-Powered GUI代理对Fine-Print注入的脆弱性 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.

摘要: 由大型语言模型（LLM）驱动的图形用户界面代理是一个专门的自治系统，根据高级指令代表用户执行任务。它通过感知和解释相关应用程序的图形用户界面（GUIs）（通常是视觉上的），推断必要的操作序列，然后通过执行单击、打字和点击等操作与GUIs交互来实现这一目标。为了完成现实世界的任务，例如填写表格或预订服务，图形用户界面代理通常需要处理和处理敏感用户数据。然而，这种自主性带来了新的隐私和安全风险。对手可以将恶意内容注入图形用户界面，从而改变代理行为或导致私人信息的意外泄露。这些攻击通常利用代理和人类用户的视觉显著性之间的差异，或者代理检测任务自动化中上下文完整性违规的能力有限。在本文中，我们描述了六种类型的此类攻击，并进行了一项实验研究，使用六个最先进的图形用户界面代理、234个对抗性网页和39名人类参与者来测试这些攻击。我们的研究结果表明，图形用户界面代理非常容易受到攻击，特别是对于上下文嵌入式威胁。此外，人类用户也容易受到许多此类攻击，这表明简单的人类监督可能无法可靠地防止故障。这种错位凸显了隐私感知代理设计的必要性。我们提出了实用的防御策略，为开发更安全、更可靠的图形用户界面代理提供信息。



## **17. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

Slice+Slice Baby：眨眼间生成末级缓存驱逐集 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11208v1) [paper-pdf](http://arxiv.org/pdf/2504.11208v1)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.

摘要: 发起缓存攻击的一个重要步骤是找到驱逐集，即争夺缓存空间的内存位置集合。在英特尔处理器上，识别竞争地址的主要挑战之一是切片高速缓存设计，其中处理器对物理地址进行哈希处理以确定内存位置存储在高速缓存中的位置。虽然过去的工作已经证明哈希函数可以颠倒，但他们也表明它取决于对手不知道的物理地址位。   在这部作品中，我们对寻找驱逐集的艺术做出了三项主要贡献。我们首先利用微体系结构的竞争比较内存访问时间，并确定地址映射到的缓存片。然后，我们使用已知的散列函数，以减少我们的切片识别方法中的错误率，并通过将切片映射外推到未经测试的内存地址来减少工作。最后，我们展示了如何传播信息驱逐集在不同的页面偏移量的非线性散列函数的情况下，迄今为止尚未探索。   我们的贡献允许在Intel i7 - 9850H上在0.7秒内生成整个LLC驱逐集，在i9 - 10900K上在1.6秒内生成整个LLC驱逐集，两者都使用非线性函数。与最先进的技术相比，这是一个显着的改进，分别耗时9倍和10倍。



## **18. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

R-TPT：通过测试时提示调优提高视觉语言模型的对抗鲁棒性 cs.LG

CVPR 2025

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11195v1) [paper-pdf](http://arxiv.org/pdf/2504.11195v1)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.

摘要: CLIP等视觉语言模型（VLM）作为基础模型已受到广泛欢迎，并开发了多种微调方法来增强下游任务的性能。然而，由于其固有的脆弱性以及从有限的开源模型集中进行选择的常见做法，VLM比传统视觉模型面临更高的对抗攻击风险。现有的防御技术通常依赖于训练期间的对抗微调，这需要标记数据并且缺乏下游任务的灵活性。为了解决这些限制，我们提出了鲁棒的测试时即时调优（R-TPT），它可以减轻推理阶段对抗性攻击的影响。我们首先通过消除在对抗条件下引入冲突的术语来重新制定经典的边际熵目标，只保留逐点的熵最小化。此外，我们引入了一种即插即用的、基于可靠性的加权集成策略，该策略从可靠的增强视图中聚合有用信息以加强防御。R-TPT增强了对对抗攻击的防御，而不需要标记的训练数据，同时为推理任务提供高度灵活性。对广泛使用的具有各种攻击的基准进行了大量实验，证明了R-TPT的有效性。该代码可在https://github.com/TomSheng21/R-TPT上找到。



## **19. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

探索LLM授权建议的后门攻击和防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.

摘要: 大型语言模型（LLM）与推荐系统（RecSys）的融合极大地提高了个性化推荐并引起了广泛关注。尽管取得了令人印象深刻的进展，但基于LLM的RecSys抵御后门攻击的安全性在很大程度上仍然没有得到充分的探索。在本文中，我们提出了一个新问题：具有特定触发器的后门是否会被注入到基于LLM的Recsys中，从而导致当后门触发器附加到项目标题时推荐响应的操纵？为了调查基于LLM的RecSys在后门攻击下的漏洞，我们提出了一种新的攻击框架，称为RecSys后门注入中毒（BadRec）。BadRec通过触发器扰乱这些物品的标题，并雇用几名虚假用户与这些物品互动，有效地毒害了训练集，并为基于LLM的RecSys注入后门。全面的实验表明，仅用对抗性示例毒害1%的训练数据就足以成功植入后门，从而能够操纵推荐。为了进一步减轻此类安全威胁，我们提出了一种名为毒药扫描仪（P-Scanner）的通用防御策略。具体来说，我们引入了基于LLM的毒物扫描仪，通过利用LLM强大的语言理解能力和丰富的知识来检测有毒物品。触发增强代理被用来生成不同的合成触发器，以引导中毒扫描器学习中毒物品检测任务的特定于领域的知识。在三个真实数据集上的大量实验验证了所提出的P-Scanner的有效性。



## **20. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

安全的文本到图像生成：简单地消除提示嵌入 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.10329v2) [paper-pdf](http://arxiv.org/pdf/2411.10329v2)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to some extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a core requirement of T2I models, we identify that textual representations are likely the primary source of unsafe generation. To this end, we propose Embedding Sanitizer (ES), which enhances the safety of T2I models by sanitizing inappropriate concepts in prompt embeddings. To our knowledge, ES is the first interpretable safe generation framework that assigns a score to each token in the prompt to indicate its potential harmfulness. In addition, ES adopts a plug-and-play modular design, offering compatibility for seamless integration with various T2I models and other safeguards. Evaluations on five prompt benchmarks show that ES outperforms eleven existing safeguard baselines, achieving state-of-the-art robustness while maintaining high-quality image generation.

摘要: 近年来，文本到图像（T2 I）生成模型在生成与文本描述一致的高质量图像方面取得了重大进展。然而，这些模型还面临着不安全生成的风险，可能会产生违反使用政策的有害内容，例如显式材料。现有的安全生成方法通常专注于通过从视觉表示中删除不需要的概念来抑制不适当的内容，同时忽视对文本表示的净化。尽管这些方法在一定程度上有助于降低滥用的风险，但在处理对抗性攻击时，它们的稳健性仍然不足。   鉴于输入文本和输出图像之间的语义一致性是T2 I模型的核心要求，我们发现文本表示可能是不安全生成的主要来源。为此，我们提出了嵌入Sanitizer（ES），它通过在提示嵌入中清理不适当的概念来增强T2 I模型的安全性。据我们所知，ES是第一个可解释的安全生成框架，它为提示中的每个令牌分配一个分数，以指示其潜在的危害性。此外，ES采用即插即用的模块化设计，提供与各种T2 I型号和其他保障措施的无缝集成的兼容性。对五个即时基准的评估表明，ES优于十一个现有的保障基线，实现了最先进的鲁棒性，同时保持高质量的图像生成。



## **21. Helper-Friendly Latency-Bounded Mitigation Strategies against Reactive Jamming Adversaries**

针对反应性干扰对手的助手友好的、有延迟限制的缓解策略 cs.IT

16 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11110v1) [paper-pdf](http://arxiv.org/pdf/2504.11110v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: Due to the recent developments in the field of full-duplex radios and cognitive radios, a new class of reactive jamming attacks has gained attention wherein an adversary transmits jamming energy over the victim's frequency band and also monitors various energy statistics in the network so as to detect countermeasures, thereby trapping the victim. Although cooperative mitigation strategies against such security threats exist, they are known to incur spectral-efficiency loss on the helper node, and are also not robust to variable latency-constraints on victim's messages. Identifying these research gaps in existing countermeasures against reactive jamming attacks, we propose a family of helper-friendly cooperative mitigation strategies that are applicable for a wide-range of latency-requirements on the victim's messages as well as practical radio hardware at the helper nodes. The proposed strategies are designed to facilitate reliable communication for the victim, without compromising the helper's spectral efficiency and also minimally disturbing the various energy statistics in the network. For theoretical guarantees on their efficacy, interesting optimization problems are formulated on the choice of the underlying parameters, followed by extensive mathematical analyses on their error-performance and covertness. Experimental results indicate that the proposed strategies should be preferred over the state-of-the-art methods when the helper node is unwilling to compromise on its error performance for assisting the victim.

摘要: 由于目前在全频无线电和认知无线电领域的发展，一类新型反应性干扰攻击引起了人们的关注，其中对手在受害者的频段上传输干扰能量，并监控网络中的各种能量统计数据，以检测对抗措施，从而陷阱受害者。尽管存在针对此类安全威胁的合作缓解策略，但众所周知，它们会在帮助节点上导致频谱效率损失，并且对受害者消息的可变延迟约束也不鲁棒。通过识别针对反应性干扰攻击的现有对策中的这些研究差距，我们提出了一系列对助手友好的合作缓解策略，这些策略适用于受害者消息的广泛延迟要求以及助手节点处的实用无线电硬件。所提出的策略旨在促进受害者的可靠通信，而不损害助手的频谱效率，并且最大限度地干扰网络中的各种能量统计数据。为了从理论上保证其功效，在基本参数的选择上制定了有趣的优化问题，然后对其错误性能和隐蔽性进行了广泛的数学分析。实验结果表明，当帮助节点不愿意牺牲其帮助受害者的错误性能时，所提出的策略应该优于最先进的方法。



## **22. Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models**

越狱文本到图像模型的令牌级约束边界搜索 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11106v1) [paper-pdf](http://arxiv.org/pdf/2504.11106v1)

**Authors**: Jiangtao Liu, Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Recent advancements in Text-to-Image (T2I) generation have significantly enhanced the realism and creativity of generated images. However, such powerful generative capabilities pose risks related to the production of inappropriate or harmful content. Existing defense mechanisms, including prompt checkers and post-hoc image checkers, are vulnerable to sophisticated adversarial attacks. In this work, we propose TCBS-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. By iteratively optimizing tokens near these boundaries, TCBS-Attack generates semantically coherent adversarial prompts capable of bypassing multiple defensive layers in T2I models. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 45\% and an ASR-1 of 21\% on jailbreaking full-chain T2I models, significantly surpassing baseline methods.

摘要: 文本到图像（T2 I）生成的最新进展显着增强了生成图像的真实感和创造力。然而，如此强大的生成能力带来了与制作不适当或有害内容相关的风险。现有的防御机制，包括提示检查器和事后图像检查器，很容易受到复杂的对抗攻击。在这项工作中，我们提出了TCBS-Attack，这是一种新型的基于查询的黑匣子越狱攻击，可以搜索位于文本和图像检查器定义的决策边界附近的令牌。通过迭代优化这些边界附近的令牌，TCBS-Attack生成语义一致的对抗性提示，能够绕过T2 I模型中的多个防御层。大量的实验表明，我们的方法在各种T2 I模型中始终优于最先进的越狱攻击，包括安全训练的开源模型和商业在线服务，如DALL-E 3。TCBS-Attack在越狱全链T2 I模型上实现了45%的ASR-4和21%的ASR-1，大大超过了基线方法。



## **23. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

揭露欺诈团伙对图神经网络的威胁：针对基于GNN的欺诈检测器的多目标图注入攻击 cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.18370v3) [paper-pdf](http://arxiv.org/pdf/2412.18370v3)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.

摘要: 图神经网络（GNN）已经成为欺诈检测、识别欺诈用户和发现恶意行为的有效工具。然而，对基于GNN的欺诈检测器的攻击及其风险很少被研究，从而使潜在的威胁未得到解决。最近的调查结果表明，欺诈越来越多地以团伙或集团的形式组织起来。在这项工作中，我们设计的攻击场景中，欺诈团伙的目标是使他们的欺诈节点被错误地归类为良性的勾结，通过掩盖他们的非法活动。基于这些场景，我们通过模拟三个现实世界的欺诈案件（垃圾邮件评论、假新闻和医疗保险欺诈）中欺诈团伙的攻击，研究针对基于GNN的欺诈检测器的对抗攻击。我们将这些攻击定义为多目标图注入攻击，并提出MonTi，一种基于变换器的多目标一次性图注入攻击模型。MonTi通过Transformer编码器同时生成所有攻击节点的属性和边，比大多数按顺序生成这些元素的现有图注入攻击方法更有效地捕获属性和边之间的相互依赖性。此外，MonTi自适应地为每个攻击节点分配度预算，以探索涉及目标、候选和攻击节点的不同注入结构，而不像现有方法那样固定所有攻击节点的度预算。实验表明，MonTi在五个现实世界的图上的表现优于最先进的图注入攻击方法。



## **24. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIIR：基于互信息的对抗鲁棒性的掩蔽图像建模 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2312.04960v4) [paper-pdf](http://arxiv.org/pdf/2312.04960v4)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism.

摘要: 视觉变形者（ViT）已成为一种基本架构，并成为现代视觉语言模型的支柱。尽管ViT的性能令人印象深刻，但其对规避攻击表现出明显的脆弱性，因此需要开发针对其独特架构定制的专门对抗训练（AT）策略。虽然直接的解决方案可能涉及将现有的AT方法应用于ViT，但我们的分析揭示了显着的不兼容性，特别是与最先进的（SOTA）方法，例如Generalist（CVPR 2023）和DBAT（USENIX Security 2024）。本文对ViT中的对抗鲁棒性进行了系统研究，并在其基于自动编码器的自我监督预训练中提供了一种新颖的理论互信息（MI）分析。具体来说，我们表明，对抗性的例子和它的潜在的表示在基于ViT的自动编码器之间的MI应通过派生的MI边界的约束。基于这一见解，我们提出了一种自监督AT方法MIMIR，该方法采用MI惩罚，通过使用自动编码器进行掩码图像建模来促进对抗性预训练。在CIFAR-10、Tiny-ImageNet和ImageNet-1 K上的大量实验表明，MIMIR可以始终如一地提供更好的自然和鲁棒的准确性，其中MIMIR在ImageNet-1 K上的结果优于SOTA AT。值得注意的是，MIMIR对不可预见的攻击和常见的损坏数据表现出卓越的鲁棒性，并且还可以承受对手完全了解防御机制的自适应攻击。



## **25. RAID: An In-Training Defense against Attribute Inference Attacks in Recommender Systems**

RAGE：推荐系统中针对属性推理攻击的训练中防御 cs.IR

17 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11510v1) [paper-pdf](http://arxiv.org/pdf/2504.11510v1)

**Authors**: Xiaohua Feng, Yuyuan Li, Fengyuan Yu, Ke Xiong, Junjie Fang, Li Zhang, Tianyu Du, Chaochao Chen

**Abstract**: In various networks and mobile applications, users are highly susceptible to attribute inference attacks, with particularly prevalent occurrences in recommender systems. Attackers exploit partially exposed user profiles in recommendation models, such as user embeddings, to infer private attributes of target users, such as gender and political views. The goal of defenders is to mitigate the effectiveness of these attacks while maintaining recommendation performance. Most existing defense methods, such as differential privacy and attribute unlearning, focus on post-training settings, which limits their capability of utilizing training data to preserve recommendation performance. Although adversarial training extends defenses to in-training settings, it often struggles with convergence due to unstable training processes. In this paper, we propose RAID, an in-training defense method against attribute inference attacks in recommender systems. In addition to the recommendation objective, we define a defensive objective to ensure that the distribution of protected attributes becomes independent of class labels, making users indistinguishable from attribute inference attacks. Specifically, this defensive objective aims to solve a constrained Wasserstein barycenter problem to identify the centroid distribution that makes the attribute indistinguishable while complying with recommendation performance constraints. To optimize our proposed objective, we use optimal transport to align users with the centroid distribution. We conduct extensive experiments on four real-world datasets to evaluate RAID. The experimental results validate the effectiveness of RAID and demonstrate its significant superiority over existing methods in multiple aspects.

摘要: 在各种网络和移动应用程序中，用户极易受到属性推断攻击，其中推荐系统中尤其普遍。攻击者利用推荐模型中部分暴露的用户配置文件（例如用户嵌入）来推断目标用户的私人属性，例如性别和政治观点。防御者的目标是降低这些攻击的有效性，同时保持推荐性能。大多数现有的防御方法，例如差异隐私和属性取消学习，都集中在训练后设置上，这限制了它们利用训练数据来保持推荐性能的能力。尽管对抗性训练将防御扩展到训练中的环境，但由于训练过程不稳定，它经常难以收敛。在本文中，我们提出了一种针对推荐系统中属性推理攻击的训练中防御方法--RAIDER。除了推荐目标之外，我们还定义了一个防御目标，以确保受保护属性的分布独立于类标签，使用户无法与属性推断攻击区分开来。具体来说，这个防御目标旨在解决受约束的Wasserstein重心问题，以识别使属性不可区分的重心分布，同时遵守推荐性能约束。为了优化我们提出的目标，我们使用最佳传输来使用户与重心分布保持一致。我们对四个现实世界的数据集进行了广泛的实验来评估磁盘阵列。实验结果验证了磁盘阵列的有效性，并展示了其在多个方面相对于现有方法的显着优势。



## **26. Inferring Communities of Interest in Collaborative Learning-based Recommender Systems**

基于协作学习的推荐系统中的兴趣社区推断 cs.IR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2306.08929v3) [paper-pdf](http://arxiv.org/pdf/2306.08929v3)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems, such as those employing Federated Learning (FL) and Gossip Learning (GL), allow users to train models while keeping their history of liked items on their devices. While these methods were seen as promising for enhancing privacy, recent research has shown that collaborative learning can be vulnerable to various privacy attacks. In this paper, we propose a novel attack called Community Inference Attack (CIA), which enables an adversary to identify community members based on a set of target items. What sets CIA apart is its efficiency: it operates at low computational cost by eliminating the need for training surrogate models. Instead, it uses a comparison-based approach, inferring sensitive information by comparing users' models rather than targeting any specific individual model. To evaluate the effectiveness of CIA, we conduct experiments on three real-world recommendation datasets using two recommendation models under both Federated and Gossip-like settings. The results demonstrate that CIA can be up to 10 times more accurate than random guessing. Additionally, we evaluate two mitigation strategies: Differentially Private Stochastic Gradient Descent (DP-SGD) and a Share less policy, which involves sharing fewer, less sensitive model parameters. Our findings suggest that the Share less strategy offers a better privacy-utility trade-off, especially in GL.

摘要: 基于协作学习的推荐系统，例如采用联邦学习（FL）和Gossip Learning（GL）的推荐系统，允许用户训练模型，同时在设备上保留喜欢项目的历史记录。虽然这些方法被认为有希望增强隐私，但最近的研究表明，协作学习可能容易受到各种隐私攻击。在本文中，我们提出了一种名为社区推理攻击（CIA）的新型攻击，它使对手能够根据一组目标项识别社区成员。CIA的与众不同之处在于其效率：它通过消除训练代理模型的需要，以较低的计算成本运行。相反，它使用基于比较的方法，通过比较用户的模型而不是针对任何特定的个体模型来推断敏感信息。为了评估CIA的有效性，我们在联邦和类似Gossip的环境下使用两种推荐模型对三个现实世界的推荐数据集进行了实验。结果表明，CIA的准确性比随机猜测高出10倍。此外，我们还评估了两种缓解策略：差异私人随机梯度下降（DP-BCD）和共享较少政策，该政策涉及共享更少、更不敏感的模型参数。我们的研究结果表明，少分享策略提供了更好的隐私与公用事业权衡，尤其是在GL中。



## **27. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

QAVA：对大型视觉语言模型的查询不可知视觉攻击 cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.

摘要: 在典型的多模式任务中，例如视觉问题解答（VQA），针对特定图像和问题的对抗攻击可能会导致大型视觉语言模型（LVLM）提供错误的答案。然而，单个图像与多个问题关联是常见的，即使对于受到特定问题攻击的对抗图像，LVLM仍然可以正确回答其他问题。为了解决这个问题，我们引入了查询不可知视觉攻击（QAVA），其目的是创建强大的对抗性示例，这些示例会对未指定和未知的问题生成错误的响应。与针对特定图像和问题的传统对抗攻击相比，QAVA显着增强了问题未知时图像攻击的有效性和效率，实现了与针对已知目标问题的攻击相当的性能。我们的研究扩大了实际环境中对LVLM的视觉对抗攻击的范围，揭示了以前被忽视的漏洞，特别是在视觉对抗威胁的背景下。该代码可在https://github.com/btzyd/qava上获取。



## **28. Defending Against Frequency-Based Attacks with Diffusion Models**

利用扩散模型防御基于频率的攻击 cs.CV

Conference on Computer Vision and Pattern Recognition Workshops  (CVPRW), 5th Workshop on Adversarial Machine Learning in Computer Vision:  Foundation Models + X

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11034v1) [paper-pdf](http://arxiv.org/pdf/2504.11034v1)

**Authors**: Fatemeh Amerehi, Patrick Healy

**Abstract**: Adversarial training is a common strategy for enhancing model robustness against adversarial attacks. However, it is typically tailored to the specific attack types it is trained on, limiting its ability to generalize to unseen threat models. Adversarial purification offers an alternative by leveraging a generative model to remove perturbations before classification. Since the purifier is trained independently of both the classifier and the threat models, it is better equipped to handle previously unseen attack scenarios. Diffusion models have proven highly effective for noise purification, not only in countering pixel-wise adversarial perturbations but also in addressing non-adversarial data shifts. In this study, we broaden the focus beyond pixel-wise robustness to explore the extent to which purification can mitigate both spectral and spatial adversarial attacks. Our findings highlight its effectiveness in handling diverse distortion patterns across low- to high-frequency regions.

摘要: 对抗训练是增强模型针对对抗攻击的稳健性的常见策略。然而，它通常是针对其训练的特定攻击类型进行定制的，从而限制了其概括为不可见威胁模型的能力。对抗性纯化提供了一种替代方案，通过利用生成模型在分类之前去除扰动。由于净化器是独立于分类器和威胁模型进行训练的，因此它能够更好地处理以前未见过的攻击场景。事实证明，扩散模型对于噪音净化非常有效，不仅可以对抗像素级对抗性扰动，而且可以解决非对抗性数据漂移。在这项研究中，我们将焦点扩大到像素稳健性之外，以探索净化可以在多大程度上减轻光谱和空间对抗攻击。我们的研究结果强调了它在处理低频到高频区域的各种失真模式方面的有效性。



## **29. RF Sensing Security and Malicious Exploitation: A Comprehensive Survey**

RF传感安全和恶意利用：全面调查 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10969v1) [paper-pdf](http://arxiv.org/pdf/2504.10969v1)

**Authors**: Mingda Han, Huanqi Yang, Wenhao Li, Weitao Xu, Xiuzhen Cheng, Prasant Mohapatra, Pengfei Hu

**Abstract**: Radio Frequency (RF) sensing technologies have experienced significant growth due to the widespread adoption of RF devices and the Internet of Things (IoT). These technologies enable numerous applications across healthcare, smart homes, industrial automation, and human-computer interaction. However, the non-intrusive and ubiquitous nature of RF sensing - combined with its environmental sensitivity and data dependency - makes these systems inherently vulnerable not only as attack targets, but also as powerful attack vectors. This survey presents a comprehensive analysis of RF sensing security, covering both system-level vulnerabilities - such as signal spoofing, adversarial perturbations, and model poisoning - and the misuse of sensing capabilities for attacks like cross-boundary surveillance, side-channel inference, and semantic privacy breaches. We propose unified threat models to structure these attack vectors and further conduct task-specific vulnerability assessments across key RF sensing applications, identifying their unique attack surfaces and risk profiles. In addition, we systematically review defense strategies across system layers and threat-specific scenarios, incorporating both active and passive paradigms to provide a structured and practical view of protection mechanisms. Compared to prior surveys, our work distinguishes itself by offering a multi-dimensional classification framework based on task type, threat vector, and sensing modality, and by providing fine-grained, scenario-driven analysis that bridges theoretical models and real-world implications. This survey aims to serve as a comprehensive reference for researchers and practitioners seeking to understand, evaluate, and secure the evolving landscape of RF sensing technologies.

摘要: 由于RF设备和物联网（IOT）的广泛采用，射频（RF）传感技术经历了显着增长。这些技术实现了医疗保健、智能家居、工业自动化和人机交互领域的众多应用。然而，RF传感的非侵入性和无处不在的性质，加上其环境敏感性和数据依赖性，使得这些系统不仅作为攻击目标，而且作为强大的攻击载体，本质上容易受到攻击。这项调查对RF传感安全性进行了全面分析，涵盖系统级漏洞（例如信号欺骗、对抗性扰动和模型中毒）以及对传感能力的滥用来进行跨境监视、侧道推断和语义隐私泄露等攻击。我们提出了统一的威胁模型来构建这些攻击载体，并进一步在关键的RF传感应用中进行特定任务的漏洞评估，识别其独特的攻击表面和风险概况。此外，我们系统地审查跨系统层和特定威胁场景的防御策略，结合主动和被动范式，以提供保护机制的结构化和实用视图。与之前的调查相比，我们的工作通过提供基于任务类型、威胁载体和感知模式的多维分类框架，并提供连接理论模型和现实世界影响的细粒度、情景驱动的分析而脱颖而出。这项调查旨在为寻求了解、评估和确保射频传感技术不断变化的格局的研究人员和从业者提供全面的参考。



## **30. Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters**

具有自定义失真过滤器的对抗黑匣子攻击强化学习平台 cs.LG

Accepted at the 2025 AAAI Conference on Artificial Intelligence  Proceedings

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2501.14122v2) [paper-pdf](http://arxiv.org/pdf/2501.14122v2)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen

**Abstract**: We present a Reinforcement Learning Platform for Adversarial Black-box untargeted and targeted attacks, RLAB, that allows users to select from various distortion filters to create adversarial examples. The platform uses a Reinforcement Learning agent to add minimum distortion to input images while still causing misclassification by the target model. The agent uses a novel dual-action method to explore the input image at each step to identify sensitive regions for adding distortions while removing noises that have less impact on the target model. This dual action leads to faster and more efficient convergence of the attack. The platform can also be used to measure the robustness of image classification models against specific distortion types. Also, retraining the model with adversarial samples significantly improved robustness when evaluated on benchmark datasets. The proposed platform outperforms state-of-the-art methods in terms of the average number of queries required to cause misclassification. This advances trustworthiness with a positive social impact.

摘要: 我们提供了一个针对对抗性黑匣子无目标和有针对性攻击的强化学习平台RLAB，允许用户从各种失真过滤器中进行选择来创建对抗性示例。该平台使用强化学习代理为输入图像添加最小的失真，同时仍然导致目标模型的误分类。该代理使用一种新颖的双重作用方法在每一步探索输入图像，以识别敏感区域，以添加失真，同时去除对目标模型影响较小的噪音。这种双重动作导致攻击的收敛速度更快、更有效。该平台还可用于测量图像分类模型对特定失真类型的鲁棒性。此外，在基准数据集上评估时，使用对抗样本重新训练模型显着提高了鲁棒性。所提出的平台在导致错误分类所需的平均查询次数方面优于最先进的方法。这提高了可信度，产生了积极的社会影响。



## **31. Verifying Properties of Binary Neural Networks Using Sparse Polynomial Optimization**

使用稀疏多元优化的二元神经网络特性 cs.LG

25 pages, 2 figures, 8 tables

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2405.17049v2) [paper-pdf](http://arxiv.org/pdf/2405.17049v2)

**Authors**: Jianting Yang, Srećko Ðurašinović, Jean-Bernard Lasserre, Victor Magron, Jun Zhao

**Abstract**: This paper explores methods for verifying the properties of Binary Neural Networks (BNNs), focusing on robustness against adversarial attacks. Despite their lower computational and memory needs, BNNs, like their full-precision counterparts, are also sensitive to input perturbations. Established methods for solving this problem are predominantly based on Satisfiability Modulo Theories and Mixed-Integer Linear Programming techniques, which are characterized by NP complexity and often face scalability issues.   We introduce an alternative approach using Semidefinite Programming relaxations derived from sparse Polynomial Optimization. Our approach, compatible with continuous input space, not only mitigates numerical issues associated with floating-point calculations but also enhances verification scalability through the strategic use of tighter first-order semidefinite relaxations. We demonstrate the effectiveness of our method in verifying robustness against both $\|.\|_\infty$ and $\|.\|_2$-based adversarial attacks.

摘要: 本文探讨了验证二进制神经网络（BNN）属性的方法，重点关注针对对抗攻击的鲁棒性。尽管BNN的计算和内存需求较低，但与全精度BNN一样，BNN也对输入扰动敏感。解决该问题的既定方法主要基于可满足性模理论和混合子线性规划技术，这些技术的特点是NP复杂性，并且经常面临可扩展性问题。   我们引入了一种替代方法，该方法使用源自稀疏多边优化的半定规划松弛。我们的方法与连续输入空间兼容，不仅缓解了与浮点计算相关的数字问题，而且还通过战略性地使用更严格的一阶半定松弛来增强验证的可扩展性。我们证明了我们的方法在验证针对$\|.\|_\的稳健性方面的有效性infty$和$\|。\|_ 2美元的对抗性攻击。



## **32. CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

CDUpatch：针对双模式可见红外探测器的颜色驱动通用对抗补丁攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10888v1) [paper-pdf](http://arxiv.org/pdf/2504.10888v1)

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios.

摘要: 对抗补丁被广泛用于评估现实世界场景中物体检测系统的稳健性。这些补丁最初设计用于欺骗单模式检测器（例如，可见光或红外），并且最近已扩展到瞄准可见-红外双模式检测器。然而，现有的双模式对抗补丁攻击在不同物理场景中的攻击有效性有限。为了解决这个问题，我们提出了CDUpatch，这是一种针对跨尺度、视图和场景可见红外物体检测器的通用跨模式补丁攻击。具体来说，我们观察到颜色变化会导致不同水平的热吸收，从而导致红外成像中的温度差异。利用这一属性，我们提出了一种RGB到红外适配器，可以将RGB补丁映射到红外补丁，从而实现跨模式补丁的统一优化。通过学习对抗性斑块上的最佳颜色分布，我们可以操纵其热响应并生成对抗性红外纹理。此外，我们引入了多尺度剪裁策略并构建了一个新的可见红外数据集MSDS rone，其中包含不同尺度和视角的飞行器图像。这些数据增强策略增强了我们补丁在现实世界条件下的稳健性。对四个基准数据集进行实验（例如，DroneVehicles、LLVIP、VisDrone）表明我们的方法优于数字领域中现有的补丁攻击。广泛的物理测试进一步证实了跨规模、视图和场景的强大可移植性。



## **33. The Sword of Damocles in ViTs: Computational Redundancy Amplifies Adversarial Transferability**

ViTS中的达摩克利斯之剑：计算冗余放大了对抗性可转让性 cs.CV

Work in progress. 10 pages. 4 figures

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.10804v1) [paper-pdf](http://arxiv.org/pdf/2504.10804v1)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.

摘要: Vision Transformers（ViT）在一系列应用中表现出令人印象深刻的性能，包括许多安全关键任务。然而，它们独特的架构属性在对抗稳健性方面提出了新的挑战和机遇。特别是，我们观察到，与CNN上制作的对抗示例相比，在ViT上制作的对抗示例表现出更高的可转移性，这表明ViT包含有利于转移攻击的结构特征。在这项工作中，我们研究了计算冗余在ViT中的作用及其对对抗可转移性的影响。与之前旨在减少计算以提高效率的研究不同，我们建议利用这种冗余来提高对抗性示例的质量和可移植性。通过详细的分析，我们确定了两种形式的冗余，包括数据级和模型级，可以利用它们来放大攻击有效性。基于这一见解，我们设计了一套技术，包括注意力稀疏操作，注意力头部置换，干净的令牌正则化，幽灵MoE多样化和测试时对抗训练。ImageNet-1 k数据集上的大量实验验证了我们方法的有效性，表明我们的方法在跨不同模型架构的可移植性和通用性方面都显着优于现有基线。



## **34. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性即时蒸馏 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **35. Transfer Learning Assisted XgBoost For Adaptable Cyberattack Detection In Battery Packs**

迁移学习辅助XgBOP用于电池包中的自适应网络攻击检测 eess.SY

9 pages, 5 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10658v1) [paper-pdf](http://arxiv.org/pdf/2504.10658v1)

**Authors**: Sanchita Ghosh, Tanushree Roy

**Abstract**: Optimal charging of electric vehicle (EVs) depends heavily on reliable sensor measurements from the battery pack to the cloud-controller of the smart charging station. However, an adversary could corrupt the voltage sensor data during transmission, potentially causing local to wide-scale disruptions. Therefore, it is essential to detect sensor cyberattacks in real-time to ensure secure EV charging, and the developed algorithms must be readily adaptable to variations, including pack configurations. To tackle these challenges, we propose adaptable fine-tuning of an XgBoost-based cell-level model using limited pack-level data to use for voltage prediction and residual generation. We used battery cell and pack data from high-fidelity charging experiments in PyBaMM and `liionpack' package to train and test the detection algorithm. The algorithm's performance has been evaluated for two large-format battery packs under sensor swapping and replay attacks. The simulation results also highlight the adaptability and efficacy of our proposed detection algorithm.

摘要: 电动汽车（EV）的最佳充电在很大程度上取决于从电池组到智能充电站的云控制器的可靠传感器测量。然而，攻击者可能会在传输过程中破坏电压传感器数据，从而可能导致局部到大规模的中断。因此，实时检测传感器网络攻击以确保安全的电动汽车充电至关重要，并且开发的算法必须易于适应变化，包括电池组配置。为了解决这些挑战，我们提出了一个基于XgBoost的细胞级模型的自适应微调使用有限的包级数据用于电压预测和残差生成。我们使用来自PyBaMM和“liionpack”包中高保真充电实验的电池组和电池组数据来训练和测试检测算法。在传感器交换和回放攻击下，该算法的性能已针对两个大规格电池组进行了评估。模拟结果还凸显了我们提出的检测算法的适应性和有效性。



## **36. Secure Estimation of Battery Voltage Under Sensor Attacks: A Self-Learning Koopman Approach**

传感器攻击下电池电压的安全估计：自学习Koopman方法 eess.SY

10 pages, 5 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10639v1) [paper-pdf](http://arxiv.org/pdf/2504.10639v1)

**Authors**: Sanchita Ghosh, Tanushree Roy

**Abstract**: Cloud-based battery management system (BMS) requires accurate terminal voltage measurement data to ensure optimal and safe charging of Lithium-ion batteries. Unfortunately, an adversary can corrupt the battery terminal voltage data as it passes from the local-BMS to the cloud-BMS through the communication network, with the objective of under- or over-charging the battery. To ensure accurate terminal voltage data under such malicious sensor attacks, this paper investigates a Koopman-based secure terminal voltage estimation scheme using a two-stage error-compensated self-learning feedback. During the first stage of error correction, the potential Koopman prediction error is estimated to compensate for the error accumulation due to the linear approximation of Koopman operator. The second stage of error compensation aims to recover the error amassing from the higher-order dynamics of the Lithium-ion batteries missed by the self-learning strategy. Specifically, we have proposed two different methods for this second stage error compensation. First, an interpretable empirical correction strategy has been obtained using the open circuit voltage to state-of-charge mapping for the battery. Second, a Gaussian process regression-based data-driven method has been explored. Finally, we demonstrate the efficacy of the proposed secure estimator using both empirical and data-driven corrections.

摘要: 基于云的电池管理系统（BMC）需要准确的终端电压测量数据，以确保锂离子电池的最佳、安全充电。不幸的是，当电池端电压数据通过通信网络从本地BMC传输到云BMC时，对手可能会破坏电池端电压数据，目的是对电池充电不足或过度充电。为了确保在此类恶意传感器攻击下准确的终端电压数据，本文研究了一种基于Koopman的安全终端电压估计方案，该方案使用两阶段误差补偿自学习反馈。在误差修正的第一阶段，估计潜在的库普曼预测误差，以补偿由于库普曼运算符线性逼近而产生的误差积累。第二阶段的误差补偿旨在恢复由自学习策略错过的来自锂离子电池的高阶动力学的误差积累。具体来说，我们提出了两种不同的方法，这个第二阶段的误差补偿。首先，使用电池的开路电压到充电状态的映射，已经获得了可解释的经验校正策略。其次，探索了基于高斯过程回归的数据驱动方法。最后，我们使用经验和数据驱动修正来证明所提出的安全估计器的有效性。



## **37. Demo: ViolentUTF as An Accessible Platform for Generative AI Red Teaming**

演示：ViolentUTF作为生成性AI Red团队的可用平台 cs.CR

3 pages, 1 figure, 1 table. This is a demo paper for  CyberWarrior2025. Codes and video demo will be shared in later version of  this paper due to embargo requirements

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10603v1) [paper-pdf](http://arxiv.org/pdf/2504.10603v1)

**Authors**: Tam n. Nguyen

**Abstract**: The rapid integration of Generative AI (GenAI) into various applications necessitates robust risk management strategies which includes Red Teaming (RT) - an evaluation method for simulating adversarial attacks. Unfortunately, RT for GenAI is often hindered by technical complexity, lack of user-friendly interfaces, and inadequate reporting features. This paper introduces Violent UTF - an accessible, modular, and scalable platform for GenAI red teaming. Through intuitive interfaces (Web GUI, CLI, API, MCP) powered by LLMs and for LLMs, Violent UTF aims to empower non-technical domain experts and students alongside technical experts, facilitate comprehensive security evaluation by unifying capabilities from RT frameworks like Microsoft PyRIT, Nvidia Garak and its own specialized evaluators. ViolentUTF is being used for evaluating the robustness of a flagship LLM-based product in a large US Government department. It also demonstrates effectiveness in evaluating LLMs' cross-domain reasoning capability between cybersecurity and behavioral psychology.

摘要: 将生成式AI（GenAI）快速集成到各种应用程序中需要强大的风险管理策略，其中包括Red Teaming（RT）-一种用于模拟对抗性攻击的评估方法。不幸的是，RT for GenAI经常受到技术复杂性，缺乏用户友好界面和报告功能不足的阻碍。本文介绍Violent UTF -一个可访问的，模块化的，可扩展的GenAI红色团队平台。通过由LLM和LLM提供支持的直观界面（Web GUI，CLI，API，MCP），Violent UTF旨在为非技术领域专家和学生以及技术专家提供支持，通过统一Microsoft PyRIT，Nvidia Garak等RT框架的功能以及自己的专业评估人员来促进全面的安全评估。ViolentUTF正在用于评估美国大型政府部门基于LLM的旗舰产品的稳健性。它还展示了评估LLM网络安全和行为心理学之间跨领域推理能力的有效性。



## **38. Ctrl-Z: Controlling AI Agents via Resampling**

Ctrl-Z：通过恢复控制人工智能代理 cs.LG

bashcontrol.com

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10374v1) [paper-pdf](http://arxiv.org/pdf/2504.10374v1)

**Authors**: Aryan Bhatt, Cody Rushing, Adam Kaufman, Tyler Tracy, Vasil Georgiev, David Matolcsi, Akbir Khan, Buck Shlegeris

**Abstract**: Control evaluations measure whether monitoring and security protocols for AI systems prevent intentionally subversive AI models from causing harm. Our work presents the first control evaluation performed in an agent environment. We construct BashBench, a dataset of 257 challenging multi-step system administration tasks, and evaluate whether various safety measures can prevent an adversarially constructed AI agent from covertly downloading and executing malicious code in this environment. This multi-step setting introduces new attack and defense dynamics, which we investigate in order to design novel control protocols that prevent safety failures without hindering the ability of non-malicious agents to perform useful work. We introduce a class of control protocols called resample protocols that dynamically take additional samples of certain actions. We find these protocols significantly improve on existing techniques by selectively blocking the AI agent from executing suspicious code and incriminating the agent by generating additional examples of dangerous behavior. We measure the tradeoff between attack prevention and usefulness; our best protocol combines resampling with analysis of previous steps, reducing the success rate of attacks from 58% to 7% at a 5% cost to the performance of a non-malicious agent.

摘要: 控制评估衡量人工智能系统的监控和安全协议是否可以防止故意颠覆性人工智能模型造成伤害。我们的工作首次在代理环境中进行控制评估。我们构建BashBench，这是一个包含257个具有挑战性的多步骤系统管理任务的数据集，并评估各种安全措施是否可以防止敌对构建的人工智能代理在此环境中秘密下载和执行恶意代码。这种多步骤设置引入了新的攻击和防御动态，我们对此进行了研究，以便设计新颖的控制协议，防止安全故障，同时又不妨碍非恶意代理执行有用工作的能力。我们引入了一类称为重采样协议的控制协议，它动态地获取某些动作的额外样本。我们发现，这些协议通过选择性地阻止AI代理执行可疑代码并通过生成额外的危险行为示例来指控代理有罪，从而对现有技术进行了显着改进。我们衡量攻击预防和有用性之间的权衡;我们最好的协议将重采样与之前步骤的分析相结合，将攻击的成功率从58%降低到7%，但非恶意代理的性能损失为5%。



## **39. Universally Composable Commitments with Communicating Malicious Physically Uncloneable Functions**

具有传播恶意物理不可克隆功能的通用可组合承诺 cs.CR

12 pages + 14 pages Appendix

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10120v1) [paper-pdf](http://arxiv.org/pdf/2504.10120v1)

**Authors**: Lourenço Abecasis, Paulo Mateus, Chrysoula Vlachou

**Abstract**: In this work, we explore the possibility of universally composable (UC)-secure commitments using Physically Uncloneable Functions (PUFs) within a new adversarial model. We introduce the communicating malicious PUFs, i.e. malicious PUFs that can interact with their creator even when not in their possession, obtaining a stronger adversarial model. Prior work [ASIACRYPT 2013, LNCS, vol. 8270, pp. 100-119] proposed a compiler for constructing UC-secure commitments from ideal extractable commitments, and our task would be to adapt the ideal extractable commitment scheme proposed therein to our new model. However, we found an attack and identified a few other issues in that construction, and to address them, we modified the aforementioned ideal extractable commitment scheme and introduced new properties and tools that allow us to rigorously develop and present security proofs in this context. We propose a new UC-secure commitment scheme against adversaries that can only create stateless malicious PUFs which can receive, but not send, information from their creators. Our protocol is more efficient compared to previous proposals, as we have parallelized the ideal extractable commitments within it. The restriction to stateless malicious PUFs is significant, mainly since the protocol from [ASIACRYPT 2013, LNCS, vol. 8270, pp. 100-119] assumes malicious PUFs with unbounded state, thus limiting its applicability. However it is the only way we found to address the issues of the original construction. We hope that in future work this restriction can be lifted, and along the lines of our work, UC-secure commitments with fewer restrictions on both the state and communication can be constructed.

摘要: 在这项工作中，我们探索了在新的对抗模型中使用物理不可克隆函数（PUF）实现普遍可组合（UC）安全承诺的可能性。我们引入了通信的恶意PUF，即即使在不被其拥有的情况下也可以与其创建者互动的恶意PUF，从而获得更强的对抗模型。先前的工作[ASIACRYPT 2013，LNSO，第8270卷，第10页。100-119]提出了一个用于从理想的可提取承诺构建UC安全承诺的编译器，我们的任务是将其中提出的理想的可提取承诺方案适应我们的新模型。然而，我们发现了攻击并在该构造中识别了一些其他问题，为了解决这些问题，我们修改了上述理想的可提取承诺方案，并引入了新的属性和工具，使我们能够在此背景下严格开发和呈现安全证明。我们提出了一种新的UC安全承诺方案，针对只能创建无状态恶意PUF的对手，这些PUF可以从其创建者那里接收但不能发送信息。与之前的提案相比，我们的协议更高效，因为我们在其中并行化了理想的可提取承诺。对无状态恶意PUF的限制很重要，主要是因为[ASIACRYPT 2013，LNSO，vol. 8270，pp. 100-119]假设恶意PUF具有无界状态，从而限制了其适用性。然而，这是我们发现解决原始建筑问题的唯一方法。我们希望在未来的工作中能够取消这一限制，并按照我们的工作方针，构建UC安全承诺，对国家和通信的限制较少。



## **40. Undermining Federated Learning Accuracy in EdgeIoT via Variational Graph Auto-Encoders**

通过变分图自动编码器削弱EdgeIoT中的联邦学习准确性 cs.LG

7 pages and 6 figures. Accepted in IEEE IWCMC 2025

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10067v1) [paper-pdf](http://arxiv.org/pdf/2504.10067v1)

**Authors**: Kai Li, Shuyan Hu, Bochun Wu, Sai Zou, Wei Ni, Falko Dressler

**Abstract**: EdgeIoT represents an approach that brings together mobile edge computing with Internet of Things (IoT) devices, allowing for data processing close to the data source. Sending source data to a server is bandwidth-intensive and may compromise privacy. Instead, federated learning allows each device to upload a shared machine-learning model update with locally processed data. However, this technique, which depends on aggregating model updates from various IoT devices, is vulnerable to attacks from malicious entities that may inject harmful data into the learning process. This paper introduces a new attack method targeting federated learning in EdgeIoT, known as data-independent model manipulation attack. This attack does not rely on training data from the IoT devices but instead uses an adversarial variational graph auto-encoder (AV-GAE) to create malicious model updates by analyzing benign model updates intercepted during communication. AV-GAE identifies and exploits structural relationships between benign models and their training data features. By manipulating these structural correlations, the attack maximizes the training loss of the federated learning system, compromising its overall effectiveness.

摘要: EdgeIoT代表了一种将移动边缘计算与物联网（IoT）设备结合在一起的方法，允许在数据源附近进行数据处理。将源数据发送到服务器需要占用带宽，并且可能会损害隐私。相反，联合学习允许每个设备上传带有本地处理数据的共享机器学习模型更新。然而，这种技术依赖于聚合来自各种物联网设备的模型更新，很容易受到恶意实体的攻击，这些实体可能会将有害数据注入学习过程。本文介绍了一种针对EdgeIoT中联邦学习的新攻击方法，称为数据独立模型操作攻击。这种攻击不依赖于来自物联网设备的训练数据，而是使用对抗变分图自动编码器（AV-GAE），通过分析在通信期间拦截的良性模型更新来创建恶意模型更新。AV-GAE识别并利用良性模型及其训练数据特征之间的结构关系。通过操纵这些结构相关性，攻击最大化了联邦学习系统的训练损失，损害了其整体有效性。



## **41. SnatchML: Hijacking ML models without Training Access**

SnatchML：在没有培训访问权限的情况下劫持ML模型 cs.CR

17 pages, 14 figures, 7 tables

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2406.01708v2) [paper-pdf](http://arxiv.org/pdf/2406.01708v2)

**Authors**: Mahmoud Ghorbel, Halima Bouzidi, Ioan Marius Bilasco, Ihsen Alouani

**Abstract**: Model hijacking can cause significant accountability and security risks since the owner of a hijacked model can be framed for having their model offer illegal or unethical services. Prior works consider model hijacking as a training time attack, whereby an adversary requires full access to the ML model training. In this paper, we consider a stronger threat model for an inference-time hijacking attack, where the adversary has no access to the training phase of the victim model. Our intuition is that ML models, which are typically over-parameterized, might have the capacity to (unintentionally) learn more than the intended task they are trained for. We propose SnatchML, a new training-free model hijacking attack, that leverages the extra capacity learnt by the victim model to infer different tasks that can be semantically related or unrelated to the original one. Our results on models deployed on AWS Sagemaker showed that SnatchML can deliver high accuracy on hijacking tasks. Interestingly, while all previous approaches are limited by the number of classes in the benign task, SnatchML can hijack models for tasks that contain more classes than the original. We explore different methods to mitigate this risk; We propose meta-unlearning, which is designed to help the model unlearn a potentially malicious task while training for the original task. We also provide insights on over-parametrization as a possible inherent factor that facilitates model hijacking, and accordingly, we propose a compression-based countermeasure to counteract this attack. We believe this work offers a previously overlooked perspective on model hijacking attacks, presenting a stronger threat model and higher applicability in real-world contexts.

摘要: 模型劫持可能会导致重大的问责和安全风险，因为被劫持模型的所有者可能会因为让他们的模型提供非法或不道德的服务而受到陷害。之前的作品将模型劫持视为一种训练时间攻击，对手需要完全访问ML模型训练。在本文中，我们考虑了针对推理时劫持攻击的更强威胁模型，其中对手无法访问受害者模型的训练阶段。我们的直觉是，通常过度参数化的ML模型可能有能力（无意中）学习比其训练的预期任务更多的内容。我们提出了SnatchML，一种新的无训练模型劫持攻击，它利用受害者模型学习的额外能力来推断不同的任务，这些任务可以在语义上与原始任务相关或不相关。我们在AWS Sagemaker上部署的模型上的结果表明，SnatchML可以在劫持任务上提供高准确性。有趣的是，虽然所有以前的方法都受到良性任务中类数量的限制，但SnatchML可以劫持包含比原始任务更多类的任务模型。我们探索了不同的方法来减轻这种风险;我们提出了元学习，旨在帮助模型在为原始任务进行训练时忘记潜在的恶意任务。我们还提供了关于过度参数化作为促进模型劫持的可能内在因素的见解，因此，我们提出了一种基于压缩的对策来对抗这种攻击。我们相信这项工作提供了以前被忽视的关于模型劫持攻击的视角，在现实世界中呈现了更强大的威胁模型和更高的适用性。



## **42. Quantifying Privacy Leakage in Split Inference via Fisher-Approximated Shannon Information Analysis**

通过费舍尔逼近的香农信息分析量化分裂推理中的隐私泄露 cs.CR

17pages,14 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10016v1) [paper-pdf](http://arxiv.org/pdf/2504.10016v1)

**Authors**: Ruijun Deng, Zhihui Lu, Qiang Duan

**Abstract**: Split inference (SI) partitions deep neural networks into distributed sub-models, enabling privacy-preserving collaborative learning. Nevertheless, it remains vulnerable to Data Reconstruction Attacks (DRAs), wherein adversaries exploit exposed smashed data to reconstruct raw inputs. Despite extensive research on adversarial attack-defense games, a shortfall remains in the fundamental analysis of privacy risks. This paper establishes a theoretical framework for privacy leakage quantification using information theory, defining it as the adversary's certainty and deriving both average-case and worst-case error bounds. We introduce Fisher-approximated Shannon information (FSInfo), a novel privacy metric utilizing Fisher Information (FI) for operational privacy leakage computation. We empirically show that our privacy metric correlates well with empirical attacks and investigate some of the factors that affect privacy leakage, namely the data distribution, model size, and overfitting.

摘要: 分裂推理（SI）将深度神经网络划分为分布式子模型，从而实现保护隐私的协作学习。然而，它仍然容易受到数据重建攻击（DSA）的影响，其中对手利用暴露的破碎数据来重建原始输入。尽管对对抗性攻击-防御游戏进行了广泛的研究，但隐私风险的基本分析仍然存在缺陷。本文利用信息论建立了隐私泄露量化的理论框架，将其定义为对手的确定性，并推导出平均情况和最坏情况的错误界限。我们介绍了费舍尔近似香农信息（FSInfo），一种新的隐私度量利用费舍尔信息（FI）的操作隐私泄漏计算。我们的经验表明，我们的隐私度量与经验攻击相关，并调查了一些影响隐私泄露的因素，即数据分布，模型大小和过拟合。



## **43. Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm**

使用DeepFool算法定制深度神经网络的对抗攻击以进行目标类操纵 cs.CV

18 pages, 5 figures. Accepted by Nature Scientific Reports

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2310.13019v5) [paper-pdf](http://arxiv.org/pdf/2310.13019v5)

**Authors**: S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab, Xi Xiao, Sarfaraz Newaz

**Abstract**: The susceptibility of deep neural networks (DNNs) to adversarial attacks undermines their reliability across numerous applications, underscoring the necessity for an in-depth exploration of these vulnerabilities and the formulation of robust defense strategies. The DeepFool algorithm by Moosavi-Dezfooli et al. (2016) represents a pivotal step in identifying minimal perturbations required to induce misclassification of input images. Nonetheless, its generic methodology falls short in scenarios necessitating targeted interventions. Additionally, previous research studies have predominantly concentrated on the success rate of attacks without adequately addressing the consequential distortion of images, the maintenance of image quality, or the confidence threshold required for misclassification. To bridge these gaps, we introduce the Enhanced Targeted DeepFool (ET DeepFool) algorithm, an evolution of DeepFool that not only facilitates the specification of desired misclassification targets but also incorporates a configurable minimum confidence score. Our empirical investigations demonstrate the superiority of this refined approach in maintaining the integrity of images and minimizing perturbations across a variety of DNN architectures. Unlike previous iterations, such as the Targeted DeepFool by Gajjar et al. (2022), our method grants unparalleled control over the perturbation process, enabling precise manipulation of model responses. Preliminary outcomes reveal that certain models, including AlexNet and the advanced Vision Transformer, display commendable robustness to such manipulations. This discovery of varying levels of model robustness, as unveiled through our confidence level adjustments, could have far-reaching implications for the field of image recognition. Our code is available at https://github.com/FazleLabib/et_deepfool.

摘要: 深度神经网络（DNN）对对抗攻击的敏感性损害了其在众多应用程序中的可靠性，凸显了深入探索这些漏洞并制定稳健防御策略的必要性。Moosavi-Dezfooli等人（2016）的DeepFool算法是识别导致输入图像误分类所需的最小扰动的关键步骤。尽管如此，其通用方法在需要有针对性干预的情况下仍存在缺陷。此外，之前的研究主要集中在攻击的成功率上，而没有充分解决随之而来的图像失真、图像质量的维护或错误分类所需的置信阈值。为了弥合这些差距，我们引入了增强型目标DeepFool（ET DeepFool）算法，这是DeepFool的一种演变，不仅有助于规范所需的误分类目标，还包含可配置的最低置信度分数。我们的实证研究证明了这种改进方法在保持图像完整性和最大限度地减少各种DNN架构中的干扰方面的优越性。与之前的迭代（例如Gajjar等人（2022）的Target DeepFool）不同，我们的方法对扰动过程赋予了无与伦比的控制，从而能够精确操纵模型响应。初步结果显示，某些模型，包括AlexNet和先进的Vision Transformer，对此类操作表现出值得赞扬的鲁棒性。通过我们的置信水平调整揭示了不同级别的模型鲁棒性，这一发现可能对图像识别领域产生深远的影响。我们的代码可在https://github.com/FazleLabib/et_deepfool上获取。



## **44. Let the Noise Speak: Harnessing Noise for a Unified Defense Against Adversarial and Backdoor Attacks**

让噪音说话：利用噪音统一防御对抗性和后门攻击 cs.LG

20 pages, 9 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2406.13073v2) [paper-pdf](http://arxiv.org/pdf/2406.13073v2)

**Authors**: Md Hasan Shahriar, Ning Wang, Naren Ramakrishnan, Y. Thomas Hou, Wenjing Lou

**Abstract**: The exponential adoption of machine learning (ML) is propelling the world into a future of distributed and intelligent automation and data-driven solutions. However, the proliferation of malicious data manipulation attacks against ML, namely adversarial and backdoor attacks, jeopardizes its reliability in safety-critical applications. The existing detection methods are attack-specific and built upon some strong assumptions, limiting them in diverse practical scenarios. Thus, motivated by the need for a more robust, unified, and attack-agnostic defense mechanism, we first investigate the shared traits of adversarial and backdoor attacks. Based on our observation, we propose NoiSec, a reconstruction-based intrusion detection system that brings a novel perspective by shifting focus from the reconstructed input to the reconstruction noise itself, which is the foundational root cause of such malicious data alterations. NoiSec disentangles the noise from the test input, extracts the underlying features from the noise, and leverages them to recognize systematic malicious manipulation. Our comprehensive evaluation of NoiSec demonstrates its high effectiveness across various datasets, including basic objects, natural scenes, traffic signs, medical images, spectrogram-based audio data, and wireless sensing against five state-of-the-art adversarial attacks and three backdoor attacks under challenging evaluation conditions. NoiSec demonstrates strong detection performance in both white-box and black-box adversarial attack scenarios, significantly outperforming the closest baseline models, particularly in an adaptive attack setting. We will provide the code for future baseline comparison. Our code and artifacts are publicly available at https://github.com/shahriar0651/NoiSec.

摘要: 机器学习（ML）的指数级采用正在推动世界进入分布式、智能自动化和数据驱动解决方案的未来。然而，针对ML的恶意数据操纵攻击（即对抗性和后门攻击）的激增危及了其在安全关键应用程序中的可靠性。现有的检测方法是针对攻击的，并且建立在一些强有力的假设之上，从而限制了它们在不同的实际场景中。因此，出于对更强大、统一且攻击不可知的防御机制的需求，我们首先研究了对抗性攻击和后门攻击的共同特征。根据我们的观察，我们提出了NoiSec，这是一种基于重建的入侵检测系统，通过将焦点从重建输入转移到重建噪音本身，带来了一种新颖的视角，重建噪音本身是此类恶意数据更改的根本原因。NoiSec从测试输入中分离噪音，从噪音中提取底层特征，并利用它们来识别系统性恶意操纵。我们对NoiSec的全面评估证明了它在各种数据集中的高效性，包括基本对象、自然场景、交通标志、医学图像、基于谱图的音频数据和无线传感，在具有挑战性的评估条件下对抗五种最先进的对抗性攻击和三种后门攻击。NoiSec在白盒和黑盒对抗攻击场景中都表现出强大的检测性能，显着优于最接近的基线模型，特别是在自适应攻击环境中。我们将提供代码用于将来的基线比较。我们的代码和工件可在https://github.com/shahriar0651/NoiSec上公开获取。



## **45. An Investigation of Large Language Models and Their Vulnerabilities in Spam Detection**

垃圾邮件检测中的大型语言模型及其漏洞研究 cs.CR

10 pages; presented at HotSoS'2025 as a work in progress paper

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09776v1) [paper-pdf](http://arxiv.org/pdf/2504.09776v1)

**Authors**: Qiyao Tang, Xiangyang Li

**Abstract**: Spam messages continue to present significant challenges to digital users, cluttering inboxes and posing security risks. Traditional spam detection methods, including rules-based, collaborative, and machine learning approaches, struggle to keep up with the rapidly evolving tactics employed by spammers. This project studies new spam detection systems that leverage Large Language Models (LLMs) fine-tuned with spam datasets. More importantly, we want to understand how LLM-based spam detection systems perform under adversarial attacks that purposefully modify spam emails and data poisoning attacks that exploit the differences between the training data and the massages in detection, to which traditional machine learning models are shown to be vulnerable. This experimentation employs two LLM models of GPT2 and BERT and three spam datasets of Enron, LingSpam, and SMSspamCollection for extensive training and testing tasks. The results show that, while they can function as effective spam filters, the LLM models are susceptible to the adversarial and data poisoning attacks. This research provides very useful insights for future applications of LLM models for information security.

摘要: 垃圾邮件继续给数字用户带来重大挑战，使收件箱变得杂乱并构成安全风险。传统的垃圾邮件检测方法，包括基于规则的、协作的和机器学习的方法，很难跟上垃圾邮件发送者所采用的快速发展的策略。该项目研究新的垃圾邮件检测系统，该系统利用经过垃圾邮件数据集微调的大型语言模型（LLM）。更重要的是，我们想了解基于LLM的垃圾邮件检测系统在有目的地修改垃圾邮件的对抗攻击和利用检测中训练数据和按摩之间差异的数据中毒攻击下如何表现，而传统的机器学习模型被证明是脆弱的。该实验使用两种LLM模型GPT 2和BERT以及三种垃圾邮件数据集Enron、LingSpam和SMSspamCollection来执行广泛的培训和测试任务。结果表明，虽然LLM模型可以充当有效的垃圾邮件过滤器，但它们很容易受到对抗性和数据中毒攻击。这项研究为LLM模型在信息安全方面的未来应用提供了非常有用的见解。



## **46. NODE-AdvGAN: Improving the transferability and perceptual similarity of adversarial examples by dynamic-system-driven adversarial generative model**

NODE-AdvGAN：通过动态系统驱动的对抗生成模型提高对抗示例的可移植性和感知相似性 cs.LG

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2412.03539v2) [paper-pdf](http://arxiv.org/pdf/2412.03539v2)

**Authors**: Xinheng Xie, Yue Wu, Cuiyu He

**Abstract**: Understanding adversarial examples is crucial for improving model robustness, as they introduce imperceptible perturbations to deceive models. Effective adversarial examples, therefore, offer the potential to train more robust models by eliminating model singularities. We propose NODE-AdvGAN, a novel approach that treats adversarial generation as a continuous process and employs a Neural Ordinary Differential Equation (NODE) to simulate generator dynamics. By mimicking the iterative nature of traditional gradient-based methods, NODE-AdvGAN generates smoother and more precise perturbations that preserve high perceptual similarity when added to benign images. We also propose a new training strategy, NODE-AdvGAN-T, which enhances transferability in black-box attacks by tuning the noise parameters during training. Experiments demonstrate that NODE-AdvGAN and NODE-AdvGAN-T generate more effective adversarial examples that achieve higher attack success rates while preserving better perceptual quality than baseline models.

摘要: 理解对抗性示例对于提高模型稳健性至关重要，因为它们会引入难以察觉的扰动来欺骗模型。因此，有效的对抗性示例可以通过消除模型奇异性来训练更稳健的模型。我们提出了NODE-AdvGAN，这是一种新颖的方法，将对抗发电视为一个连续过程，并使用神经常微方程（NODE）来模拟发电机动态。通过模仿传统基于梯度的方法的迭代性质，NODE-AdvGAN生成更平滑、更精确的扰动，在添加到良性图像时可以保持高度的感知相似性。我们还提出了一种新的训练策略NODE-AdvGAN-T，该策略通过在训练期间调整噪音参数来增强黑匣子攻击的可转移性。实验表明，NODE-AdvGAN和NODE-AdvGAN-T生成更有效的对抗性示例，与基线模型相比，它们可以实现更高的攻击成功率，同时保留更好的感知质量。



## **47. Mitigating Many-Shot Jailbreaking**

减轻多枪越狱 cs.LG

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09604v1) [paper-pdf](http://arxiv.org/pdf/2504.09604v1)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a ``fake'' assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the ``fake'' assistant. In this work, we probe the effectiveness of different fine tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and we show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.

摘要: 多镜头越狱（MSJ）是一种对抗性技术，它利用现代LLM的长上下文窗口来规避模型安全培训，方法是在提示中包含许多“假”助手在最终请求之前做出不当反应的示例。有了足够多的例子，该模型的上下文学习能力凌驾于其安全培训之上，并且它的反应就像它是“假”助手一样。在这项工作中，我们探讨了不同的微调和输入清理方法单独和组合在减轻MSJ攻击方面的有效性。我们发现每种技术都具有增量缓解效果，并且我们表明组合的技术显着降低了MSJ攻击的有效性，同时在良性上下文学习和对话任务中保留了模型性能。我们认为，如果将我们的方法纳入模型安全培训后，可以有意义地改善这种脆弱性。



## **48. ControlNET: A Firewall for RAG-based LLM System**

Control NET：基于RAG的LLM系统的防火墙 cs.CR

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09593v1) [paper-pdf](http://arxiv.org/pdf/2504.09593v1)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.

摘要: 检索增强生成（RAG）显着增强了大型语言模型（LLM）的事实准确性和领域适应性。这一进步使它们能够在医疗保健、金融和企业应用程序等敏感领域广泛部署。RAG通过整合外部知识来缓解幻觉，但也会带来隐私风险和安全风险，尤其是数据泄露风险和数据中毒风险。虽然最近的研究探索了即时注射和中毒攻击，但在控制入站和出站查询流以减轻这些威胁的全面研究方面仍然存在显着差距。在本文中，我们提出了一种人工智能防火墙Controller NET，旨在保护基于RAG的LLM系统免受这些漏洞的影响。ControlNET通过利用激活转变现象来检测对抗性查询并通过语义分歧减轻其影响来控制查询流。我们使用最先进的开源LLM（Llama 3、Vicuna和Mistral）对四个不同的基准数据集（包括Mmarco、HotpotQA、FinQA和MedalSys）进行全面实验。我们的结果表明，ControlNET在检测和缓解安全威胁同时保持系统无害性方面达到了超过0.909 AUROC。总的来说，ControlNET提供了一种有效、健壮、无害的防御机制，标志着基于RAG的LLM系统安全部署的重大进步。



## **49. Secure Filtering against Spatio-Temporal False Data Attacks under Asynchronous Sampling**

非同步采样下的时空虚假数据攻击的安全过滤 eess.SY

9 pages and 6 figures. arXiv admin note: text overlap with  arXiv:2303.17514

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2411.19765v2) [paper-pdf](http://arxiv.org/pdf/2411.19765v2)

**Authors**: Zishuo Li, Anh Tung Nguyen, André M. H. Teixeira, Yilin Mo, Karl H. Johansson

**Abstract**: This paper addresses the secure state estimation problem for continuous linear time-invariant systems with non-periodic and asynchronous sampled measurements, where the sensors need to transmit not only measurements but also sampling time-stamps to the fusion center. This measurement and communication setup is well-suited for operating large-scale control systems and, at the same time, introduces new vulnerabilities that can be exploited by adversaries through (i) manipulation of measurements, (ii) manipulation of time-stamps, (iii) elimination of measurements, (iv) generation of completely new false measurements, or a combination of these attacks. To mitigate these attacks, we propose a decentralized estimation algorithm in which each sensor maintains its local state estimate asynchronously based on its measurements. The local states are synchronized through time prediction and fused after time-stamp alignment. In the absence of attacks, state estimates are proven to recover the optimal Kalman estimates by solving a weighted least square problem. In the presence of attacks, solving this weighted least square problem with the aid of $\ell_1$ regularization provides secure state estimates with uniformly bounded error under an observability redundancy assumption. The effectiveness of the proposed algorithm is demonstrated using a benchmark example of the IEEE 14-bus system.

摘要: 本文讨论了具有非周期性和同步采样测量的连续线性时不变系统的安全状态估计问题，其中传感器不仅需要将测量结果传输到融合中心，还需要将采样时间戳传输到融合中心。这种测量和通信设置非常适合操作大规模控制系统，同时引入了新的漏洞，对手可以通过以下方式利用这些漏洞：（i）操作测量，（ii）操作时间戳，（iii）消除测量，（iv）生成全新的错误测量，或这些攻击的组合。为了减轻这些攻击，我们提出了一种去中心化的估计算法，其中每个传感器根据其测量值同步维护其局部状态估计。局部状态通过时间预测进行同步，并在时间戳对齐后进行融合。在没有攻击的情况下，状态估计被证明可以通过解决加权最小平方问题来恢复最佳卡尔曼估计。在存在攻击的情况下，借助$\ell_1 $正规化解决这个加权最小平方问题可以在可观测性冗余假设下提供具有一致有界误差的安全状态估计。使用IEEE 14节点系统的基准示例验证了所提出算法的有效性。



## **50. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

AdaSteer：您的对齐LLM本质上是一个自适应越狱防御者 cs.CR

17 pages, 6 figures, 9 tables

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09466v1) [paper-pdf](http://arxiv.org/pdf/2504.09466v1)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.

摘要: 尽管在安全调整方面做出了广泛的努力，但大型语言模型（LLM）仍然容易受到越狱攻击。激活转向提供了一种无需训练的防御方法，但依赖于固定的转向系数，从而导致次优保护和良性输入的错误拒绝增加。为了解决这个问题，我们提出了AdaSteer，这是一种自适应激活引导方法，可以根据输入特征动态调整模型行为。我们确定了两个关键属性：拒绝定律（R-Law），它表明与拒绝方向相反的越狱输入需要更强的引导，以及有害定律（H-Law），它区分对抗性和良性输入。AdaSteer沿着拒绝方向（RD）和有害方向（HD）引导输入表示，并通过逻辑回归学习自适应系数，确保强大的越狱防御，同时保留良性的输入处理。LLaMA-3.1、Gemma-2和Qwen 2.5的实验表明，AdaSteer在多次越狱攻击中优于基线方法，且对效用的影响最小。我们的结果强调了可解释模型内部要素在LLC中实时、灵活的安全执行方面的潜力。



