# Latest Adversarial Attack Papers
**update at 2025-06-12 10:09:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge**

LL Mail-Injects：来自现实自适应提示注入挑战的数据集 cs.CR

Dataset at:  https://huggingface.co/datasets/microsoft/llmail-inject-challenge

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09956v1) [paper-pdf](http://arxiv.org/pdf/2506.09956v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection.

摘要: 间接提示注入攻击利用大型语言模型（LLM）的固有限制来区分其输入中的指令和数据。尽管有许多防御提案，但针对自适应对手的系统评估仍然有限，即使成功的攻击可能会产生广泛的安全和隐私影响，并且许多基于现实世界的LLM应用程序仍然容易受到攻击。我们展示了LLMail-Injects的结果，这是一个模拟现实场景的公开挑战，其中参与者自适应地尝试将恶意指令注入电子邮件中，以在基于LLM的电子邮件助手中触发未经授权的工具调用。该挑战涵盖多种防御策略、LLM架构和检索配置，产生了来自839名参与者的208，095份独特攻击提交的数据集。我们发布了挑战代码、完整的提交数据集以及我们的分析，展示了这些数据如何为描述-数据分离问题提供新的见解。我们希望这将成为未来研究的基础，以推动注入的实用结构性解决方案。



## **2. Generate-then-Verify: Reconstructing Data from Limited Published Statistics**

生成然后验证：从有限的已发布统计数据重建数据 stat.ML

First two authors contributed equally. Remaining authors are ordered  alphabetically

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.21199v2) [paper-pdf](http://arxiv.org/pdf/2504.21199v2)

**Authors**: Terrance Liu, Eileen Xiao, Adam Smith, Pratiksha Thaker, Zhiwei Steven Wu

**Abstract**: We study the problem of reconstructing tabular data from aggregate statistics, in which the attacker aims to identify interesting claims about the sensitive data that can be verified with 100% certainty given the aggregates. Successful attempts in prior work have conducted studies in settings where the set of published statistics is rich enough that entire datasets can be reconstructed with certainty. In our work, we instead focus on the regime where many possible datasets match the published statistics, making it impossible to reconstruct the entire private dataset perfectly (i.e., when approaches in prior work fail). We propose the problem of partial data reconstruction, in which the goal of the adversary is to instead output a $\textit{subset}$ of rows and/or columns that are $\textit{guaranteed to be correct}$. We introduce a novel integer programming approach that first $\textbf{generates}$ a set of claims and then $\textbf{verifies}$ whether each claim holds for all possible datasets consistent with the published aggregates. We evaluate our approach on the housing-level microdata from the U.S. Decennial Census release, demonstrating that privacy violations can still persist even when information published about such data is relatively sparse.

摘要: 我们研究从聚合统计数据重建表格数据的问题，其中攻击者的目标是识别有关敏感数据的有趣声明，这些声明可以在给定聚合物的情况下以100%的确定性进行验证。之前的工作中的成功尝试是在已发布的统计数据集足够丰富的环境中进行的研究，以至于可以确定地重建整个数据集。在我们的工作中，我们专注于许多可能的数据集与已发布的统计数据相匹配的制度，这使得不可能完美地重建整个私人数据集（即，当之前工作中的方法失败时）。我们提出了部分数据重建的问题，其中对手的目标是输出$\texit {subset}$的行和/或列，$\texit {保证是正确的}$。我们引入了一种新的整数规划方法，首先$\textbf{生成}$一组索赔，然后$\textbf{验证}$是否每个索赔持有所有可能的数据集一致的发布的聚合。我们评估了我们对美国十年一次的人口普查发布的住房层面微观数据的方法，表明即使发布的有关此类数据的信息相对较少，侵犯隐私的行为仍然存在。



## **3. Apollo: A Posteriori Label-Only Membership Inference Attack Towards Machine Unlearning**

Apollo：针对机器取消学习的后验纯标签成员推理攻击 cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09923v1) [paper-pdf](http://arxiv.org/pdf/2506.09923v1)

**Authors**: Liou Tang, James Joshi, Ashish Kundu

**Abstract**: Machine Unlearning (MU) aims to update Machine Learning (ML) models following requests to remove training samples and their influences on a trained model efficiently without retraining the original ML model from scratch. While MU itself has been employed to provide privacy protection and regulatory compliance, it can also increase the attack surface of the model. Existing privacy inference attacks towards MU that aim to infer properties of the unlearned set rely on the weaker threat model that assumes the attacker has access to both the unlearned model and the original model, limiting their feasibility toward real-life scenarios. We propose a novel privacy attack, A Posteriori Label-Only Membership Inference Attack towards MU, Apollo, that infers whether a data sample has been unlearned, following a strict threat model where an adversary has access to the label-output of the unlearned model only. We demonstrate that our proposed attack, while requiring less access to the target model compared to previous attacks, can achieve relatively high precision on the membership status of the unlearned samples.

摘要: 机器非学习（MU）旨在根据请求更新机器学习（ML）模型，以有效地删除训练样本及其对训练模型的影响，而无需从头重新训练原始ML模型。虽然MU本身被用来提供隐私保护和监管合规性，但它也会增加模型的攻击面。现有的针对MU的隐私推断攻击旨在推断未学习集的属性，依赖于较弱的威胁模型，该模型假设攻击者可以访问未学习的模型和原始模型，从而限制了其在现实生活场景中的可行性。我们提出了一种新颖的隐私攻击，即针对MU、Apollo的后验标签成员资格推断攻击，它遵循严格的威胁模型，其中对手只能访问未学习的模型的标签输出。我们证明，与之前的攻击相比，我们提出的攻击虽然需要更少的访问目标模型，但可以对未学习样本的成员身份状态实现相对高的精确度。



## **4. A look at adversarial attacks on radio waveforms from discrete latent space**

从离散潜在空间看无线电波的对抗性攻击 cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09896v1) [paper-pdf](http://arxiv.org/pdf/2506.09896v1)

**Authors**: Attanasia Garuso, Silvija Kokalj-Filipovic, Yagna Kaasaragadda

**Abstract**: Having designed a VQVAE that maps digital radio waveforms into discrete latent space, and yields a perfectly classifiable reconstruction of the original data, we here analyze the attack suppressing properties of VQVAE when an adversarial attack is performed on high-SNR radio-frequency (RF) data-points. To target amplitude modulations from a subset of digitally modulated waveform classes, we first create adversarial attacks that preserve the phase between the in-phase and quadrature component whose values are adversarially changed. We compare them with adversarial attacks of the same intensity where phase is not preserved. We test the classification accuracy of such adversarial examples on a classifier trained to deliver 100% accuracy on the original data. To assess the ability of VQVAE to suppress the strength of the attack, we evaluate the classifier accuracy on the reconstructions by VQVAE of the adversarial datapoints and show that VQVAE substantially decreases the effectiveness of the attack. We also compare the I/Q plane diagram of the attacked data, their reconstructions and the original data. Finally, using multiple methods and metrics, we compare the probability distribution of the VQVAE latent space with and without attack. Varying the attack strength, we observe interesting properties of the discrete space, which may help detect the attacks.

摘要: 设计了一个VQVAE，它将数字无线电波形映射到离散潜在空间，并产生原始数据的完美可分类重建，我们在这里分析了当对高SNR射频（RF）数据点执行对抗性攻击时VQVAE的攻击抑制特性。为了针对数字调制波类子集的幅度调制，我们首先创建对抗攻击，以保留同相和正相分量之间的相，其值会发生对抗性变化。我们将它们与相同强度的对抗性攻击进行比较，其中不保留相。我们在经过训练的分类器上测试此类对抗性示例的分类准确性，该分类器可在原始数据上提供100%准确性。为了评估VQVAE抑制攻击强度的能力，我们评估了VQVAE对对抗数据点重建的分类器准确性，并表明VQVAE大大降低了攻击的有效性。我们还比较了受攻击数据的I/Q平面图、它们的重建和原始数据。最后，我们使用多种方法和指标，比较了有攻击和没有攻击的VQVAE潜在空间的概率分布。通过改变攻击强度，我们观察到离散空间的有趣属性，这可能有助于检测攻击。



## **5. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.02132v2) [paper-pdf](http://arxiv.org/pdf/2504.02132v2)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multi-modal retrieval augmented generation (M-RAG) is instrumental for inhibiting hallucinations in large multi-modal models (LMMs) through the use of a factual knowledge base (KB). However, M-RAG introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this paper, we present the first poisoning attack against M-RAG targeting visual document retrieval applications where the KB contains images of document pages. We propose two attacks, each of which require injecting only a single adversarial image into the KB. Firstly, we propose a universal attack that, for any potential user query, influences the response to cause a denial-of-service (DoS) in the M-RAG system. Secondly, we present a targeted attack against one or a group of user queries, with the goal of spreading targeted misinformation. For both attacks, we use a multi-objective gradient-based adversarial approach to craft the injected image while optimizing for both retrieval and generation. We evaluate our attacks against several visual document retrieval datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (LMMs), demonstrating the attack effectiveness in both the universal and targeted settings. We additionally present results including commonly used defenses, various attack hyper-parameter settings, ablations, and attack transferability.

摘要: 多模式检索增强生成（M-RAG）有助于通过使用事实知识库（KB）来抑制大型多模式模型（LSYS）中的幻觉。然而，M-RAG为对手引入了新的攻击载体，旨在通过将恶意条目注入知识库来破坏系统。本文中，我们提出了针对M-RAG的第一次中毒攻击，目标是KB包含文档页面图像的视觉文档检索应用程序。我们提出了两种攻击，每种攻击只需要将单个对抗图像注入到KB中。首先，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而在M-RAG系统中引起拒绝服务（DPS）。其次，我们对一个或一组用户查询进行有针对性的攻击，目标是传播有针对性的错误信息。对于这两种攻击，我们使用基于多目标梯度的对抗方法来制作注入的图像，同时优化检索和生成。我们评估了对多个视觉文档检索数据集、一组不同的最先进检索器（嵌入模型）和生成器（LSYS）的攻击，展示了在通用和目标设置中的攻击有效性。我们还提供了包括常用防御、各种攻击超参数设置、消融和攻击可转移性在内的结果。



## **6. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

CROW：通过内部一致性规范化消除大型语言模型的后门 cs.CL

Accepted at ICML 2025, 20 pages

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2411.12768v2) [paper-pdf](http://arxiv.org/pdf/2411.12768v2)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.

摘要: 大型语言模型（LLM）容易受到后门攻击，这些攻击通过隐藏触发器操纵输出。现有的防御方法（专为视觉/文本分类任务设计）无法生成文本。我们提出了内部一致性正规化（CROW），这是一种利用以下观察结果的防御，即后门模型在触发时表现出不稳定的分层隐藏表示，而干净模型则表现出平滑的过渡。CROW在微调期间通过对抗性扰动和正规化来强制跨层的一致性，中和后门，而不需要干净的参考模型或触发知识--只需一个小的干净数据集。Llama-2（7 B，13 B）、CodeLlama（7 B，13 B）和Mistral-7 B的实验证明了CROW的有效性：它在各种后门策略（情绪引导、定向拒绝、代码注入）上显着降低攻击成功率，同时保持生成性能。CROW的架构不可知设计可以实现实际部署。



## **7. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

通过交叉Wasserstein Balls进行分布和反向稳健逻辑回归 math.OC

9 main pages + 25 pages of appendices

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2407.13625v4) [paper-pdf](http://arxiv.org/pdf/2407.13625v4)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Adversarially robust optimization (ARO) has emerged as the *de facto* standard for training models that hedge against adversarial attacks in the test stage. While these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue, some successful methods replace the empirical distribution in the training stage with alternatives including *(i)* a worst-case distribution residing in an ambiguity set, resulting in a distributionally robust (DR) counterpart of ARO; *(ii)* a mixture of the empirical distribution with a distribution induced by an auxiliary (*e.g.*, synthetic, external, out-of-domain) dataset. Inspired by the former, we study the Wasserstein DR counterpart of ARO for logistic regression and show it admits a tractable convex optimization reformulation. Adopting the latter setting, we revise the DR approach by intersecting its ambiguity set with another ambiguity set built using the auxiliary dataset, which offers a significant improvement whenever the Wasserstein distance between the data generating and auxiliary distributions can be estimated. We study the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method outperforms benchmark approaches on standard datasets.

摘要: 对抗鲁棒优化（ARO）已成为在测试阶段对冲对抗攻击的训练模型的“事实上的”标准。虽然这些模型对对抗攻击很强，但它们往往会严重遭受过度匹配的影响。为了解决这个问题，一些成功的方法用替代方案取代训练阶段中的经验分布，包括 *（i）* 驻留在模糊集中的最坏情况分布，从而产生ARO的分布稳健（DR）对应物; *（ii）* 经验分布与由辅助（* 例如 *，合成的、外部的、域外的）数据集。受前者的启发，我们研究了ARO的Wasserstein DR逻辑回归，并表明它允许易于处理的凸优化重新公式化。采用后一种设置，我们通过将其模糊度集与使用辅助数据集构建的另一个模糊度集相交来修改DR方法，每当可以估计数据生成和辅助分布之间的Wasserstein距离时，这都会提供显着的改进。我们研究潜在的优化问题，开发有效的解决方案算法，并证明所提出的方法优于标准数据集的基准方法。



## **8. Evasion Attacks Against Bayesian Predictive Models**

针对Bayesian预测模型的规避攻击 stat.ML

Accepted as an oral presentation at UAI'25

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09640v1) [paper-pdf](http://arxiv.org/pdf/2506.09640v1)

**Authors**: Pablo G. Arce, Roi Naveiro, David Ríos Insua

**Abstract**: There is an increasing interest in analyzing the behavior of machine learning systems against adversarial attacks. However, most of the research in adversarial machine learning has focused on studying weaknesses against evasion or poisoning attacks to predictive models in classical setups, with the susceptibility of Bayesian predictive models to attacks remaining underexplored. This paper introduces a general methodology for designing optimal evasion attacks against such models. We investigate two adversarial objectives: perturbing specific point predictions and altering the entire posterior predictive distribution. For both scenarios, we propose novel gradient-based attacks and study their implementation and properties in various computational setups.

摘要: 人们对分析机器学习系统对抗对抗攻击的行为越来越感兴趣。然而，对抗性机器学习的大部分研究都集中在研究经典设置中针对预测模型的规避或毒害攻击的弱点，而Bayesian预测模型对攻击的易感性仍然没有得到充分的研究。本文介绍了一种设计针对此类模型的最佳规避攻击的通用方法。我们研究了两个对抗目标：扰乱特定点预测和改变整个后验预测分布。对于这两种情况，我们提出了新型的基于梯度的攻击，并研究了它们在各种计算设置中的实现和属性。



## **9. Effective Red-Teaming of Policy-Adherent Agents**

有效的政策遵守人员红色团队 cs.MA

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09600v1) [paper-pdf](http://arxiv.org/pdf/2506.09600v1)

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks

摘要: 以任务为导向的基于LLM的代理越来越多地用于具有严格政策（例如退款资格或取消规则）的领域。挑战在于确保代理始终遵守这些规则和政策，适当拒绝任何违反规则和政策的请求，同时仍然保持有用且自然的交互。这需要开发量身定制的设计和评估方法，以确保代理针对恶意用户行为的弹性。我们提出了一种新颖的威胁模型，重点关注旨在利用遵守政策的代理来谋取个人利益的对抗用户。为了解决这个问题，我们提出了CRAFT，这是一个多代理红色团队系统，它利用政策感知的说服策略来破坏客户服务场景中遵守政策的代理，优于传统的越狱方法，例如DAN提示、情绪操纵和胁迫。在现有的tau-table基准的基础上，我们引入了tau-break，这是一个补充基准，旨在严格评估代理针对操纵用户行为的稳健性。最后，我们评估了几种简单但有效的防御策略。虽然这些措施提供了一些保护，但它们还不够，这凸显了需要更强大的、以研究为驱动的保障措施来保护遵守政策的代理人免受对抗性攻击



## **10. TooBadRL: Trigger Optimization to Boost Effectiveness of Backdoor Attacks on Deep Reinforcement Learning**

TooBadRL：触发优化以提高后门攻击对深度强化学习的有效性 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09562v1) [paper-pdf](http://arxiv.org/pdf/2506.09562v1)

**Authors**: Songze Li, Mingxuan Zhang, Oubo Ma, Kang Wei, Shouling Ji

**Abstract**: Deep reinforcement learning (DRL) has achieved remarkable success in a wide range of sequential decision-making domains, including robotics, healthcare, smart grids, and finance. Recent research demonstrates that attackers can efficiently exploit system vulnerabilities during the training phase to execute backdoor attacks, producing malicious actions when specific trigger patterns are present in the state observations. However, most existing backdoor attacks rely primarily on simplistic and heuristic trigger configurations, overlooking the potential efficacy of trigger optimization. To address this gap, we introduce TooBadRL (Trigger Optimization to Boost Effectiveness of Backdoor Attacks on DRL), the first framework to systematically optimize DRL backdoor triggers along three critical axes, i.e., temporal, spatial, and magnitude. Specifically, we first introduce a performance-aware adaptive freezing mechanism for injection timing. Then, we formulate dimension selection as a cooperative game, utilizing Shapley value analysis to identify the most influential state variable for the injection dimension. Furthermore, we propose a gradient-based adversarial procedure to optimize the injection magnitude under environment constraints. Evaluations on three mainstream DRL algorithms and nine benchmark tasks show that TooBadRL significantly improves attack success rates, while ensuring minimal degradation of normal task performance. These results highlight the previously underappreciated importance of principled trigger optimization in DRL backdoor attacks. The source code of TooBadRL can be found at https://github.com/S3IC-Lab/TooBadRL.

摘要: 深度强化学习（DRL）在广泛的顺序决策领域取得了巨大的成功，包括机器人、医疗保健、智能电网和金融。最近的研究表明，攻击者可以在训练阶段有效地利用系统漏洞来执行后门攻击，当状态观测中存在特定的触发模式时，就会产生恶意行为。然而，大多数现有的后门攻击主要依赖于简单化和启发式的触发器配置，忽略了触发器优化的潜在功效。为了解决这个问题，我们引入了TooBadRL（触发器优化以提高DRL后门攻击的有效性），这是第一个沿着三个关键轴系统优化DRL后门触发器的框架，即，时间、空间和幅度。具体来说，我们首先引入了一种用于注射定时的性能感知自适应冻结机制。然后，我们将维度选择制定为合作博弈，利用Shapley值分析来识别对注入维度最有影响力的状态变量。此外，我们提出了一种基于梯度的对抗程序来优化环境约束下的注入量。对三种主流DRL算法和九个基准任务的评估表明，TooBadRL显着提高了攻击成功率，同时确保正常任务性能的下降最小。这些结果凸显了DRL后门攻击中原则性触发优化的重要性之前被低估。TooBadRL的源代码可在https://github.com/S3IC-Lab/TooBadRL上找到。



## **11. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.07736v2) [paper-pdf](http://arxiv.org/pdf/2506.07736v2)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **12. AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches**

AngleRoCL：针对物理观点不变T2 I对抗补丁的角度稳健概念学习 cs.CV

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09538v1) [paper-pdf](http://arxiv.org/pdf/2506.09538v1)

**Authors**: Wenjun Ji, Yuxiang Fu, Luyang Ying, Deng-Ping Fan, Yuyi Wang, Ming-Ming Cheng, Ivor Tsang, Qing Guo

**Abstract**: Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents.

摘要: 最前沿的作品表明，文本到图像（T2 I）扩散模型可以生成对抗补丁，误导物理世界中最先进的对象检测器，揭示检测器的漏洞和风险。然而，当从物理世界的不同角度观察时，这些方法忽视了T2 I补丁的攻击有效性（即，T2 I对抗补丁的角度稳健性）。本文全面研究了T2 I对抗补丁的角度鲁棒性，揭示了它们的角度鲁棒性问题，证明文本对生成补丁的角度鲁棒性有显着影响，而特定任务的语言指令未能增强角度鲁棒性。受这些研究的启发，我们引入了角度稳健概念学习（AngleRoCL），这是一种简单灵活的方法，可以学习可概括的概念（即，实现中的文本嵌入）表示生成角度稳健补丁的能力。学习到的概念可以被整合到文本提示中，并引导T2 I模型生成攻击有效性本质上可以抵抗观点变化的补丁。通过对多个视图的五个SOTA检测器进行广泛的模拟和物理世界实验，我们证明与基线方法相比，AngleRoCL显着增强了T2 I对抗斑块的角度稳健性。即使在具有挑战性的观看条件下，我们的补丁也能保持很高的攻击成功率，多个角度的攻击有效性平均相对提高超过50%。这项研究促进了对物理角度稳健补丁的理解，并深入了解T2 I生成的内容中的文本概念和物理属性之间的关系。



## **13. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

尖峰神经网络的隐私风险：成员推断分析 cs.LG

14 pages, 6 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2502.13191v4) [paper-pdf](http://arxiv.org/pdf/2502.13191v4)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://github.com/sharmaabhijith/MIA_SNN.

摘要: 尖峰神经网络（SNN）在现实世界应用中因其能源效率和鲁棒性而受到越来越多的探索，但其隐私风险在很大程度上仍未得到审查。在这项工作中，我们调查了SNN对成员推断攻击（MIA）的敏感性--这是一种主要的隐私威胁，对手试图确定给定样本是否是训练数据集的一部分。虽然之前的工作表明SNN由于其离散的、事件驱动的性质而可能提供固有的鲁棒性，但我们发现其弹性随着延迟（T）的增加而减弱。此外，我们在黑匣子设置下引入了输入丢弃策略，这显着增强了SNN中的成员推断。我们的研究结果挑战了SNN本质上更安全的假设，尽管它们预计会更好，但我们的结果表明SNN表现出与人工神经网络（ANN）同等可比的隐私漏洞。我们的代码可在https://github.com/sharmaabhijith/MIA_SNN上获取。



## **14. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09443v1) [paper-pdf](http://arxiv.org/pdf/2506.09443v1)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的智能，这激发了LLM作为法官系统的开发和广泛采用，用于自动化模型测试，例如红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发人们对其稳健性的担忧，从而对其可信度。LLM法官采用的现有评估方法往往是零碎的，缺乏统一的综合评估框架。此外，很少探索用于提高判断稳健性的提示模板和模型选择，而且它们在现实世界环境中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。RobustJudge调查攻击方法和防御策略的影响（MQ 1），探索提示模板和模型选择的影响（MQ 2），并评估现实世界的LLM作为法官应用程序的稳健性（MQ 3）。我们的主要发现是：（1）法学硕士作为法官系统仍然容易受到一系列对抗攻击，包括联合攻击和PAIR，而重新标记化和基于LLM的检测器等防御机制提供了更好的保护;（2）鲁棒性对提示模板和判断模型的选择高度敏感。我们提出的提示模板优化方法可以提高稳健性，JudggeLM-13 B作为稳健的开源法官表现出了强大的性能;（3）将RobustJudge应用于阿里巴巴的PRI平台，揭示了之前未报告的漏洞。RobustJudge的源代码可访问https://github.com/S3IC-Lab/RobustJudge。



## **15. AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)**

对抗性漏洞质量指数（AVQI）：通过GRACE保护LLM：几何表示-感知对比增强-引入对抗性漏洞质量指数（AVQI） cs.CL

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08885v2) [paper-pdf](http://arxiv.org/pdf/2506.08885v2)

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.

摘要: 针对LLM的对抗威胁升级的速度超出了当前防御系统的适应能力。我们暴露了对齐中的一个关键几何盲点：对抗性提示利用潜在伪装，危险地嵌入到靠近安全表示集合的地方，同时编码不安全的意图，从而规避直接偏好优化（DPO）等表面级别的防御，这些防御仍然对潜在的几何形状视而不见。我们引入了ALKARI，这是第一个经过严格策划的对抗基准，也是迄今为止最全面的基准，涵盖三个宏类别、六个亚型和十五个攻击家族的9，000个提示。对21个领先LLM的评估显示，开放源和封闭源模型的攻击成功率（ASB）都高得惊人，暴露了我们称之为“潜在伪装”的潜在漏洞，这是一个结构盲点，对抗性完成模仿安全的潜在几何形状。为了缓解这一漏洞，我们引入了GRACE -几何表示感知对比增强，这是一个将偏好学习与潜在空间正规化结合起来的对齐框架。GRACE强制执行两个限制：安全完成和对抗完成之间的潜在分离，以及不安全和越狱行为之间的对抗凝聚力。这些在学习注意力配置文件的指导下通过分层池嵌入进行操作，在不修改基本模型的情况下重塑内部几何形状，并实现高达39%的ASB降低。此外，我们还引入了AVQI，这是一种几何感知指标，通过集群分离和紧凑性量化潜在的对齐失败。AVQI揭示了不安全完工何时模仿安全完工的几何形状，为模型如何内部编码安全性提供了原则性的视角。我们在https://anonymous.4open.science/r/alkali-B416/README.md上公开该代码。



## **16. Adversarial Surrogate Risk Bounds for Binary Classification**

二元分类的对抗性代理风险界限 cs.LG

37 pages, 2 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09348v1) [paper-pdf](http://arxiv.org/pdf/2506.09348v1)

**Authors**: Natalie S. Frank

**Abstract**: A central concern in classification is the vulnerability of machine learning models to adversarial attacks. Adversarial training is one of the most popular techniques for training robust classifiers, which involves minimizing an adversarial surrogate risk. Recent work characterized when a minimizing sequence of an adversarial surrogate risk is also a minimizing sequence of the adversarial classification risk for binary classification -- a property known as adversarial consistency. However, these results do not address the rate at which the adversarial classification risk converges to its optimal value for such a sequence of functions that minimize the adversarial surrogate. This paper provides surrogate risk bounds that quantify that convergence rate. Additionally, we derive distribution-dependent surrogate risk bounds in the standard (non-adversarial) learning setting, that may be of independent interest.

摘要: 分类中的一个核心问题是机器学习模型对对抗攻击的脆弱性。对抗性训练是训练稳健分类器最流行的技术之一，它涉及最大限度地减少对抗性代理风险。最近的工作的特点是，对抗性替代风险的最小化序列也是二元分类的对抗性分类风险的最小化序列--这一属性称为对抗性一致性。然而，这些结果并没有解决对抗性分类风险收敛到最佳值的速度，以使对抗性替代物最小化。本文提供了量化收敛率的替代风险界限。此外，我们在标准（非对抗性）学习环境中推导出依赖于分布的替代风险界限，这可能具有独立的兴趣。



## **17. PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**

PatchGuard：通过视觉变换器和伪异常进行逆向鲁棒异常检测和定位 cs.CV

Accepted to the Conference on Computer Vision and Pattern Recognition  (CVPR) 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.09237v1) [paper-pdf](http://arxiv.org/pdf/2506.09237v1)

**Authors**: Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban

**Abstract**: Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .

摘要: 异常检测（AD）和异常定位（AL）在医学成像和工业监控等要求高可靠性的领域至关重要。然而，由于训练数据的限制，当前的AD和AL方法通常容易受到对抗攻击，训练数据通常只包括正常的、未标记的样本。本研究引入了PatchGuard，这是一种对抗稳健的AD和AL方法，它在基于Vision Transformer（ViT）的架构中将伪异常与定位屏蔽结合起来，以解决这些漏洞。我们首先研究伪异常的基本属性，然后提供增强AD和AL系统对抗鲁棒性所需的注意机制的理论见解。然后，我们介绍了我们的方法，该方法利用前景感知伪异常来克服之前异常感知方法的缺陷。我们的方法将这些精心设计的伪异常样本整合到基于ViT的框架中，并在旨在提高模型稳健性的新型损失函数指导下进行对抗训练，正如我们的理论分析所支持的那样。在成熟的工业和医疗数据集上的实验结果表明，PatchGuard在对抗环境中的表现显着优于之前的方法，在AD中实现了53.2%美元的性能收益，在AL中实现了68.5%美元的性能收益，同时在非对抗环境中还保持了有竞争力的准确性。代码存储库可在https://github.com/rohban-lab/PatchGuard上获取。



## **18. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

PEFTGuard：检测后门攻击对抗参数高效微调 cs.CR

21 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.17453v2) [paper-pdf](http://arxiv.org/pdf/2411.17453v2)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters. To fill this gap, we first construct and release PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision that our benchmark and method can shed light on future LLM backdoor detection research.

摘要: 微调是提高特定领域大型语言模型（LLM）性能的重要过程，参数高效微调（PEFT）因其能够通过集成低级适配器来减少计算需求而越来越受欢迎。这些轻量级适配器（例如LoRA）可以在开源平台上共享和使用。然而，对手可能会利用这种机制向这些适配器注入后门，导致不正确或有害输出等恶意行为，从而给社区带来严重的安全风险。不幸的是，目前很少有工作专注于分析后门模式或检测适配器中的后门。为了填补这一空白，我们首先构建并发布PADBench，这是一个全面的基准测试，包含13，300个良性和后门适配器，经过各种数据集、攻击策略、PEFT方法和LLM微调。此外，我们提出了PEFTGuard，第一个后门检测框架对基于PEFT的适配器。对PADBench的广泛评估表明，PEFTGuard优于现有的检测方法，在大多数情况下实现了近乎完美的检测准确率（100%）。值得注意的是，PEFTGuard在三个方面表现出零射击可移植性，包括不同的攻击，PEFT方法和适配器等级。此外，我们还考虑各种自适应攻击来证明PEFTGuard的高稳健性。我们进一步探索了几种可能的后门缓解防御措施，发现精细混合是最有效的方法。我们设想我们的基准和方法可以为未来的LLM后门检测研究提供线索。



## **19. Unified Breakdown Analysis for Byzantine Robust Gossip**

拜占庭鲁棒流言的统一分解分析 math.OC

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2410.10418v3) [paper-pdf](http://arxiv.org/pdf/2410.10418v3)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: In decentralized machine learning, different devices communicate in a peer-to-peer manner to collaboratively learn from each other's data. Such approaches are vulnerable to misbehaving (or Byzantine) devices. We introduce F-RG, a general framework for building robust decentralized algorithms with guarantees arising from robust-sum-like aggregation rules F. We then investigate the notion of *breakdown point*, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce a practical robust aggregation rule, coined CS+, such that CS+-RG has a near-optimal breakdown. Other choices of aggregation rules lead to existing algorithms such as ClippedGossip or NNA. We give experimental evidence to validate the effectiveness of CS+-RG and highlight the gap with NNA, in particular against a novel attack tailored to decentralized communications.

摘要: 在去中心化机器学习中，不同的设备以点对点的方式进行通信，以协作地从彼此的数据中学习。此类方法很容易受到行为不当（或拜占庭式）设备的影响。我们引入了F-RG，这是一个用于构建稳健去中心化算法的通用框架，其保证源自稳健和类聚合规则F。然后，我们研究 * 崩溃点 * 的概念，并给出去中心化算法可以容忍的对手数量的上限。我们引入了一个实用的鲁棒聚合规则，即CS+，这样CS+-RG就有了接近最优的分解。聚合规则的其他选择导致现有算法，例如ClipedGossip或NNA。我们提供了实验证据来验证CS+-RG的有效性，并强调了与NNA的差距，特别是针对去中心化通信量身定制的新型攻击。



## **20. Adversarial Text Generation with Dynamic Contextual Perturbation**

动态上下文扰动的对抗性文本生成 cs.CR

This is the accepted version of the paper, which was presented at  IEEE CALCON. The conference was organized at Jadavpur University, Kolkata,  from December 14 to 15, 2025. The paper is six pages long, and it consists of  six tables and six figures. This is not the final camera-ready version of the  paper

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.09148v1) [paper-pdf](http://arxiv.org/pdf/2506.09148v1)

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit, Subhasis Dasgupta

**Abstract**: Adversarial attacks on Natural Language Processing (NLP) models expose vulnerabilities by introducing subtle perturbations to input text, often leading to misclassification while maintaining human readability. Existing methods typically focus on word-level or local text segment alterations, overlooking the broader context, which results in detectable or semantically inconsistent perturbations. We propose a novel adversarial text attack scheme named Dynamic Contextual Perturbation (DCP). DCP dynamically generates context-aware perturbations across sentences, paragraphs, and documents, ensuring semantic fidelity and fluency. Leveraging the capabilities of pre-trained language models, DCP iteratively refines perturbations through an adversarial objective function that balances the dual objectives of inducing model misclassification and preserving the naturalness of the text. This comprehensive approach allows DCP to produce more sophisticated and effective adversarial examples that better mimic natural language patterns. Our experimental results, conducted on various NLP models and datasets, demonstrate the efficacy of DCP in challenging the robustness of state-of-the-art NLP systems. By integrating dynamic contextual analysis, DCP significantly enhances the subtlety and impact of adversarial attacks. This study highlights the critical role of context in adversarial attacks and lays the groundwork for creating more robust NLP systems capable of withstanding sophisticated adversarial strategies.

摘要: 对自然语言处理（NLP）模型的对抗攻击通过对输入文本引入微妙的扰动来暴露漏洞，通常会导致错误分类，同时保持人类的可读性。现有的方法通常专注于单词级或本地文本片段的改变，而忽略了更广泛的上下文，这会导致可检测到或语义不一致的扰动。我们提出了一种新型的对抗性文本攻击方案，名为动态上下文扰动（DPP）。DPP动态地生成句子、段落和文档之间的上下文感知扰动，确保语义保真度和流畅性。利用预先训练的语言模型的能力，DPP通过对抗性目标函数迭代地细化扰动，该函数平衡了引发模型错误分类和保持文本自然性的双重目标。这种全面的方法使DPP能够生成更复杂、更有效的对抗示例，从而更好地模拟自然语言模式。我们在各种NLP模型和数据集上进行的实验结果证明了DPP在挑战最先进NLP系统稳健性方面的功效。通过集成动态上下文分析，DPP显着增强了对抗性攻击的微妙性和影响力。这项研究强调了上下文在对抗性攻击中的关键作用，并为创建能够承受复杂对抗策略的更强大的NLP系统奠定了基础。



## **21. Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing**

通过随机平滑的可证明成本敏感的对抗性防御 cs.LG

Published in ICML 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2310.08732v3) [paper-pdf](http://arxiv.org/pdf/2310.08732v3)

**Authors**: Yuan Xin, Dingfan Chen, Michael Backes, Xiao Zhang

**Abstract**: As ML models are increasingly deployed in critical applications, robustness against adversarial perturbations is crucial. While numerous defenses have been proposed to counter such attacks, they typically assume that all adversarial transformations are equally important, an assumption that rarely aligns with real-world applications. To address this, we study the problem of robust learning against adversarial perturbations under cost-sensitive scenarios, where the potential harm of different types of misclassifications is encoded in a cost matrix. Our solution introduces a provably robust learning algorithm to certify and optimize for cost-sensitive robustness, building on the scalable certification framework of randomized smoothing. Specifically, we formalize the definition of cost-sensitive certified radius and propose our novel adaptation of the standard certification algorithm to generate tight robustness certificates tailored to any cost matrix. In addition, we design a robust training method that improves certified cost-sensitive robustness without compromising model accuracy. Extensive experiments on benchmark datasets, including challenging ones unsolvable by existing methods, demonstrate the effectiveness of our certification algorithm and training method across various cost-sensitive scenarios.

摘要: 随着ML模型越来越多地部署在关键应用中，针对对抗性扰动的鲁棒性至关重要。虽然已经提出了许多防御措施来对抗此类攻击，但它们通常假设所有对抗转换都同样重要，这一假设很少与现实世界的应用程序相一致。为了解决这个问题，我们研究了成本敏感场景下针对对抗性扰动的鲁棒学习问题，其中不同类型错误分类的潜在危害被编码在成本矩阵中。我们的解决方案引入了一种可证明稳健的学习算法，以随机平滑的可扩展认证框架为基础，验证和优化成本敏感的稳健性。具体来说，我们正式化了成本敏感认证半径的定义，并提出了对标准认证算法的新颖调整，以生成针对任何成本矩阵定制的严格稳健性证书。此外，我们设计了一种稳健的训练方法，可以在不影响模型准确性的情况下提高经过认证的成本敏感稳健性。对基准数据集的广泛实验，包括现有方法无法解决的具有挑战性的数据集，证明了我们的认证算法和训练方法在各种成本敏感场景中的有效性。



## **22. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

Pre-print

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2412.07192v2) [paper-pdf](http://arxiv.org/pdf/2412.07192v2)

**Authors**: Zachary Coalson, Jeonghyun Woo, Yu Sun, Shiyang Chen, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们对商业规模（与人类一致的）语言模型引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来引发越狱。我们的对手可以通过在所有情况下少于25个位翻转来越狱数十亿参数的语言模型，在某些情况下只需只需5个位翻转，比对计算机视觉模型的现有攻击少40美元\x $，至少小100美元\x $。与基于预算的越狱不同，我们的攻击使内存中的这些模型在运行时“未经审查”，使它们能够在无需任何输入修改的情况下生成有害响应。我们的攻击算法有效地识别要翻转的目标位，比之前的方法提供高达20美元\x $的计算效率。这使得具有数十亿个参数的语言模型变得实用。我们展示了使用软件诱导的故障注入Rowhammer（RH）对攻击的端到端利用。我们的工作检查了具有不同RH漏洞的DDR4和LPDDR 4X设备的56个RAM RH配置文件。我们表明，我们的攻击可以可靠地在与受先前位翻转攻击影响的系统类似的系统中引发越狱。此外，即使针对高度RH安全的系统（例如，比之前测试的系统安全46 $\x $）。我们的分析进一步揭示了：（1）训练后对齐较少的模型需要更少的位翻转即可越狱;（2）某些模型组件，例如价值投影层，比其他组件更容易受到攻击;（3）我们的方法在机械上与现有的越狱不同。我们的研究结果凸显了语言模型生态系统面临的紧迫、实际威胁，并强调了研究以保护这些模型免受位翻转攻击的必要性。



## **23. Towards Robust Deep Reinforcement Learning against Environmental State Perturbation**

迈向对抗环境状态扰动的稳健深度强化学习 cs.LG

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08961v1) [paper-pdf](http://arxiv.org/pdf/2506.08961v1)

**Authors**: Chenxu Wang, Huaping Liu

**Abstract**: Adversarial attacks and robustness in Deep Reinforcement Learning (DRL) have been widely studied in various threat models; however, few consider environmental state perturbations, which are natural in embodied scenarios. To improve the robustness of DRL agents, we formulate the problem of environmental state perturbation, introducing a preliminary non-targeted attack method as a calibration adversary, and then propose a defense framework, named Boosted Adversarial Training (BAT), which first tunes the agents via supervised learning to avoid catastrophic failure and subsequently adversarially trains the agent with reinforcement learning. Extensive experimental results substantiate the vulnerability of mainstream agents under environmental state perturbations and the effectiveness of our proposed attack. The defense results demonstrate that while existing robust reinforcement learning algorithms may not be suitable, our BAT framework can significantly enhance the robustness of agents against environmental state perturbations across various situations.

摘要: 深度强化学习（DRL）中的对抗性攻击和鲁棒性已在各种威胁模型中得到广泛研究;然而，很少有人考虑环境状态扰动，这在具体场景中是自然的。为了提高DRL代理的鲁棒性，我们提出了环境状态扰动问题，引入了初步的非目标攻击方法作为校准对手，然后提出了一个名为加强对抗训练（BAT）的防御框架，该框架首先通过监督学习调整代理以避免灾难性失败，然后通过强化学习对抗训练代理。大量的实验结果证实了主流代理在环境状态扰动下的脆弱性以及我们提出的攻击的有效性。辩护结果表明，虽然现有的鲁棒强化学习算法可能不适合，但我们的BAT框架可以显着增强代理在各种情况下对环境状态扰动的鲁棒性。



## **24. Towards a Re-evaluation of Data Forging Attacks in Practice**

重新评估实践中的数据伪造攻击 cs.CR

18 pages

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.05658v2) [paper-pdf](http://arxiv.org/pdf/2411.05658v2)

**Authors**: Mohamed Suliman, Anisa Halimi, Swanand Kadhe, Nathalie Baracaldo, Douglas Leith

**Abstract**: Data forging attacks provide counterfactual proof that a model was trained on a given dataset, when in fact, it was trained on another. These attacks work by forging (replacing) mini-batches with ones containing distinct training examples that produce nearly identical gradients. Data forging appears to break any potential avenues for data governance, as adversarial model owners may forge their training set from a dataset that is not compliant to one that is. Given these serious implications on data auditing and compliance, we critically analyse data forging from both a practical and theoretical point of view, finding that a key practical limitation of current attack methods makes them easily detectable by a verifier; namely that they cannot produce sufficiently identical gradients. Theoretically, we analyse the question of whether two distinct mini-batches can produce the same gradient. Generally, we find that while there may exist an infinite number of distinct mini-batches with real-valued training examples and labels that produce the same gradient, finding those that are within the allowed domain e.g. pixel values between 0-255 and one hot labels is a non trivial task. Our results call for the reevaluation of the strength of existing attacks, and for additional research into successful data forging, given the serious consequences it may have on machine learning and privacy.

摘要: 数据伪造攻击提供了反事实证据，证明模型是在给定数据集上训练的，而事实上，它是在另一个数据集上训练的。这些攻击的工作原理是用包含产生几乎相同梯度的不同训练示例的小批量伪造（替换）小批量。数据伪造似乎打破了数据治理的任何潜在途径，因为对抗模型所有者可能会从不兼容数据集的数据集中伪造他们的训练集。鉴于这些对数据审计和合规性的严重影响，我们从实践和理论的角度批判性地分析了数据伪造，发现当前攻击方法的一个关键实践限制使它们很容易被验证者检测到;即它们无法产生足够相同的梯度。从理论上讲，我们分析了两个不同的迷你批次是否可以产生相同梯度的问题。一般来说，我们发现，虽然可能存在无限多个具有产生相同梯度的实值训练示例和标签的不同小批量，但找到那些在允许的域内的小批量，例如0-255之间的像素值和一个热标签是一项不平凡的任务。鉴于数据伪造可能对机器学习和隐私产生严重后果，我们的研究结果呼吁重新评估现有攻击的强度，并对成功的数据伪造进行额外研究。



## **25. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

以毒攻毒（F3）：LVLM中一种无需培训且高效的视觉对抗示例净化方法 cs.CV

14 pages, 5 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.01064v2) [paper-pdf](http://arxiv.org/pdf/2506.01064v2)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available.

摘要: 大型视觉语言模型（LVLM）的最新进展展示了它们在广泛的多模式视觉语言任务中的非凡能力。然而，这些模型仍然容易受到视觉对抗攻击，这可能会极大地损害其性能。尽管它们具有潜在的影响，但净化此类对抗性例子的有效方法的开发受到的关注相对有限。在本文中，我们介绍了F3，这是一个新颖的对抗净化框架，它采用了违反直觉的“以毒攻毒”策略：有意地向对抗性示例引入简单的扰动以减轻其有害影响。具体来说，F3利用从随机干扰的对手示例中获得的跨模式注意力作为参考目标。通过向这些对抗性示例中注入噪音，F3有效地细化了他们的注意力，从而产生更干净、更可靠的模型输出。值得注意的是，这种看似矛盾的利用噪音来抵消对抗攻击的方法产生了令人印象深刻的净化结果。此外，F3具有几个明显的优势：无需训练且易于实施，并且与现有的纯化方法相比，计算效率显着提高。这些属性使F3特别适合大规模工业应用，其中稳健的性能和运营效率都是关键优先事项。该代码将公开。



## **26. Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks**

基于Lipschitz有界网络的高效鲁棒共形预测 cs.LG

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.05434v2) [paper-pdf](http://arxiv.org/pdf/2506.05434v2)

**Authors**: Thomas Massena, Léo andéol, Thibaut Boissin, Franck Mamalet, Corentin Friedrich, Mathieu Serrurier, Sébastien Gerchinovitz

**Abstract**: Conformal Prediction (CP) has proven to be an effective post-hoc method for improving the trustworthiness of neural networks by providing prediction sets with finite-sample guarantees. However, under adversarial attacks, classical conformal guarantees do not hold anymore: this problem is addressed in the field of Robust Conformal Prediction. Several methods have been proposed to provide robust CP sets with guarantees under adversarial perturbations, but, for large scale problems, these sets are either too large or the methods are too computationally demanding to be deployed in real life scenarios. In this work, we propose a new method that leverages Lipschitz-bounded networks to precisely and efficiently estimate robust CP sets. When combined with a 1-Lipschitz robust network, we demonstrate that our lip-rcp method outperforms state-of-the-art results in both the size of the robust CP sets and computational efficiency in medium and large-scale scenarios such as ImageNet. Taking a different angle, we also study vanilla CP under attack, and derive new worst-case coverage bounds of vanilla CP sets, which are valid simultaneously for all adversarial attack levels. Our lip-rcp method makes this second approach as efficient as vanilla CP while also allowing robustness guarantees.

摘要: 事实证明，共形预测（CP）是一种有效的事后方法，可以通过提供具有有限样本保证的预测集来提高神经网络的可信度。然而，在对抗性攻击下，经典的保形保证不再成立：这个问题在鲁棒保形预测领域得到解决。已经提出了几种方法来提供具有对抗性扰动下保证的稳健CP集，但是，对于大规模问题，这些集要么太大，要么这些方法的计算要求太高，无法部署在现实生活场景中。在这项工作中，我们提出了一种新方法，利用Lipschitz有界网络来精确有效地估计稳健CP集。当与1-Lipschitz稳健网络结合时，我们证明了我们的lip-rcp方法在稳健CP集的大小和中型和大型场景（例如ImageNet）中的计算效率方面都优于最先进的结果。我们还从不同的角度研究了攻击下的vanilla CP，并推导出vanilla CP集的新的最坏情况覆盖界限，该界限对所有对抗攻击级别同时有效。我们的lip-rcp方法使第二种方法与vanilla CP一样高效，同时还可以保证稳健性。



## **27. One Patch to Rule Them All: Transforming Static Patches into Dynamic Attacks in the Physical World**

一个补丁来统治它们：将静态补丁转化为物理世界中的动态攻击 cs.CR

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08482v1) [paper-pdf](http://arxiv.org/pdf/2506.08482v1)

**Authors**: Xingshuo Han, Chen Ling, Shiyi Yao, Haozhao Wang, Hangcheng Liu, Yutong Wu, Shengmin Xu, Changhai Ou, Xinyi Huang, Tianwei Zhang

**Abstract**: Numerous methods have been proposed to generate physical adversarial patches (PAPs) against real-world machine learning systems. However, each existing PAP typically supports only a single, fixed attack goal, and switching to a different objective requires re-generating and re-deploying a new PAP. This rigidity limits their practicality in dynamic environments like autonomous driving, where traffic conditions and attack goals can change rapidly. For example, if no obstacles are present around the target vehicle, the attack may fail to cause meaningful consequences.   To overcome this limitation, we propose SwitchPatch, a novel PAP that is static yet enables dynamic and controllable attack outcomes based on real-time scenarios. Attackers can alter pre-defined conditions, e.g., by projecting different natural-color lights onto SwitchPatch to seamlessly switch between attack goals. Unlike prior work, SwitchPatch does not require re-generation or re-deployment for different objectives, significantly reducing cost and complexity. Furthermore, SwitchPatch remains benign when the enabling conditions are absent, enhancing its stealth.   We evaluate SwitchPatch on two key tasks: traffic sign recognition (classification and detection) and depth estimation. First, we conduct theoretical analysis and empirical studies to demonstrate the feasibility of SwitchPatch and explore how many goals it can support using techniques like color light projection and occlusion. Second, we perform simulation-based experiments and ablation studies to verify its effectiveness and transferability. Third, we conduct outdoor tests using a Unmanned Ground Vehicle (UGV) to confirm its robustness in the physical world. Overall, SwitchPatch introduces a flexible and practical adversarial strategy that can be adapted to diverse tasks and real-world conditions.

摘要: 已经提出了许多方法来针对现实世界的机器学习系统生成物理对抗补丁（PAP）。然而，每个现有的PAP通常只支持单个固定的攻击目标，切换到不同的目标需要重新生成和重新部署新的PAP。这种刚性限制了它们在自动驾驶等动态环境中的实用性，其中交通状况和攻击目标可能会迅速变化。例如，如果目标车辆周围没有障碍物，那么攻击可能无法造成有意义的后果。   为了克服这一局限性，我们提出了Switch patch，这是一种新型PAP，它是静态的，但可以根据实时场景实现动态且可控的攻击结果。攻击者可以改变预定义的条件，例如通过将不同的自然色灯光投射到Switch Pad上，在攻击目标之间无缝切换。与之前的工作不同，Switch patch不需要针对不同目标重新生成或重新部署，从而显着降低了成本和复杂性。此外，在缺乏使能条件时，Switch补丁仍然保持良性，增强了其隐形性。   我们评估了Switch patch的两项关键任务：交通标志识别（分类和检测）和深度估计。首先，我们进行理论分析和实证研究，以证明Switch patch的可行性，并探索它可以使用彩色光投影和遮挡等技术支持多少个目标。其次，我们进行基于模拟的实验和消融研究，以验证其有效性和可移植性。第三，我们使用无人地面车辆（UGV）进行户外测试，以确认其在物理世界中的稳健性。总的来说，Switch patch引入了一种灵活实用的对抗策略，可以适应不同的任务和现实世界的条件。



## **28. SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense**

SHIELD：用于增量扩展学习防御的安全超网络 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.08255v1) [paper-pdf](http://arxiv.org/pdf/2506.08255v1)

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek

**Abstract**: Traditional deep neural networks suffer from several limitations, including catastrophic forgetting. When models are adapted to new datasets, they tend to quickly forget previously learned knowledge. Another significant issue is the lack of robustness to even small perturbations in the input data. In practice, we can often easily perform adversarial attacks and change the network's predictions, adding minimal noise to the input. Dedicated architectures and training procedures can solve each of the above problems separately. Unfortunately, currently, no model can simultaneously address both catastrophic forgetting and vulnerability to adversarial attacks. We introduce SHIELD (Secure Hypernetworks for Incremental Expansion and Learning Defense), a novel approach that integrates a hypernetwork-based continual learning approach with interval arithmetic. SHIELD use the hypernetwork to transfer trainable task embedding vectors into the weights of a target model dedicated to specific data. This paradigm allows for the dynamic generation of separate networks for each subtask, while the hypernetwork aggregates and analyzes information across all tasks. The target model takes in the input a data sample with a defined interval range, and by creating a hypercube, produces a prediction for the given range. Therefore, such target models provide strict guarantees against all possible attacks for data samples within the interval range. Our approach enhances security without sacrificing network adaptability, addressing the overlooked challenge of safety in continual learning.

摘要: 传统的深度神经网络存在几个局限性，包括灾难性的遗忘。当模型适应新的数据集时，它们往往会很快忘记以前学到的知识。另一个重要问题是即使输入数据中很小的扰动也缺乏稳健性。在实践中，我们通常可以轻松地执行对抗攻击并改变网络的预测，从而为输入添加最小的噪音。专用的架构和培训程序可以分别解决上述每个问题。不幸的是，目前没有模型可以同时解决灾难性遗忘和对抗性攻击的脆弱性。我们介绍了SHIELD（安全超网络增量扩展和学习防御），一种新的方法，集成了基于超网络的持续学习方法与区间算术。SHIELD使用超网络将可训练的任务嵌入向量转换为专用于特定数据的目标模型的权重。该范式允许为每个子任务动态生成单独的网络，同时超网络聚合和分析所有任务的信息。目标模型接收具有定义间隔范围的数据样本的输入，并通过创建超立方体来生成对给定范围的预测。因此，此类目标模型为区间范围内的数据样本提供了严格的保证，防止所有可能的攻击。我们的方法在不牺牲网络适应性的情况下增强了安全性，解决了持续学习中被忽视的安全挑战。



## **29. Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models**

通过镜头寻找：揭示多模式大型推理模型上与位置相关的隐私泄露 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.19373v3) [paper-pdf](http://arxiv.org/pdf/2504.19373v3)

**Authors**: Weidi Luo, Tianyu Lu, Qiming Zhang, Xiaogeng Liu, Bin Hu, Yue Zhao, Jieyu Zhao, Song Gao, Patrick McDaniel, Zhen Xiang, Chaowei Xiao

**Abstract**: Recent advances in multi-modal large reasoning models (MLRMs) have shown significant ability to interpret complex visual content. While these models enable impressive reasoning capabilities, they also introduce novel and underexplored privacy risks. In this paper, we identify a novel category of privacy leakage in MLRMs: Adversaries can infer sensitive geolocation information, such as a user's home address or neighborhood, from user-generated images, including selfies captured in private settings. To formalize and evaluate these risks, we propose a three-level visual privacy risk framework that categorizes image content based on contextual sensitivity and potential for location inference. We further introduce DoxBench, a curated dataset of 500 real-world images reflecting diverse privacy scenarios. Our evaluation across 11 advanced MLRMs and MLLMs demonstrates that these models consistently outperform non-expert humans in geolocation inference and can effectively leak location-related private information. This significantly lowers the barrier for adversaries to obtain users' sensitive geolocation information. We further analyze and identify two primary factors contributing to this vulnerability: (1) MLRMs exhibit strong reasoning capabilities by leveraging visual clues in combination with their internal world knowledge; and (2) MLRMs frequently rely on privacy-related visual clues for inference without any built-in mechanisms to suppress or avoid such usage. To better understand and demonstrate real-world attack feasibility, we propose GeoMiner, a collaborative attack framework that decomposes the prediction process into two stages: clue extraction and reasoning to improve geolocation performance while introducing a novel attack perspective. Our findings highlight the urgent need to reassess inference-time privacy risks in MLRMs to better protect users' sensitive information.

摘要: 多模式大型推理模型（MLRM）的最新进展已显示出解释复杂视觉内容的强大能力。虽然这些模型具有令人印象深刻的推理能力，但它们也引入了新颖且未充分探索的隐私风险。在本文中，我们识别了MLRM中的一种新型隐私泄露类型：对手可以从用户生成的图像（包括在私人环境中拍摄的自拍照）中推断敏感的地理位置信息，例如用户的家庭住址或社区。为了正式化和评估这些风险，我们提出了一个三级视觉隐私风险框架，该框架根据上下文敏感性和位置推断的潜力对图像内容进行分类。我们进一步介绍了DoxBench，这是一个由500张现实世界图像组成的精心策划的数据集，反映了不同的隐私场景。我们对11种高级MLRM和MLLM的评估表明，这些模型在地理位置推断方面始终优于非专家人类，并且可以有效地泄露与位置相关的私人信息。这大大降低了对手获取用户敏感地理位置信息的障碍。我们进一步分析和识别了导致该漏洞的两个主要因素：（1）MLRM通过利用视觉线索与其内部世界知识相结合，展现出强大的推理能力;（2）MLRM经常依赖于与隐私相关的视觉线索进行推理，没有任何内置机制来抑制或避免此类使用。为了更好地理解和演示现实世界的攻击可行性，我们提出了GeoMiner，这是一个协作攻击框架，它将预测过程分解为两个阶段：线索提取和推理，以提高地理定位性能，同时引入新颖的攻击视角。我们的调查结果凸显了重新评估MLRM中的推断时隐私风险的迫切需要，以更好地保护用户的敏感信息。



## **30. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

代码大型语言模型的对抗性攻击分类和鲁棒性测试 cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.

摘要: 大型语言模型（LLM）已成为代码生成、完成和分析等软件开发任务的重要工具。随着它们与工作流程集成的加深，确保针对漏洞（尤其是由多样化或敌对输入触发的漏洞）的鲁棒性变得越来越重要。当模型遇到受干扰的任务描述、代码或评论时，此类漏洞可能会导致不正确或不安全的代码生成。之前的研究经常忽视自然语言在指导代码任务中的作用。本研究调查了自然语言输入（包括提示、评论和描述）中的对抗性扰动如何影响LLM for Code（LLM4Code）。它检查字符、单词和句子层面上的干扰的影响，以识别最有影响力的漏洞。我们分析了多个项目（例如，ReCode、OpenAttack）和数据集（例如，HumanEval，MBPP），建立了对抗性攻击的分类。第一个维度对输入类型代码、提示或注释进行分类，而第二个维度重点关注粒度：字符、单词或业务级别的更改。我们采用了混合方法，将定量性能指标与定性漏洞分析相结合。LLM 4Code模型显示出不同扰动类型的鲁棒性不同。句子级别的攻击效果最差，这表明模型能够适应更广泛的背景变化。相比之下，词级扰动带来了严重的挑战，暴露了语义漏洞。初级效应各不相同，表明模型对微妙的语法偏差的敏感性。我们的研究提供了一个结构化框架来测试LLM 4 Code稳健性，并强调自然语言在对抗性评估中的关键作用。提高模型对语义级中断的弹性对于安全可靠的代码生成系统至关重要。



## **31. CAPAA: Classifier-Agnostic Projector-Based Adversarial Attack**

CAPPA：基于分类不可知投影仪的对抗攻击 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.00978v2) [paper-pdf](http://arxiv.org/pdf/2506.00978v2)

**Authors**: Zhan Li, Mingyu Zhao, Xin Dong, Haibin Ling, Bingyao Huang

**Abstract**: Projector-based adversarial attack aims to project carefully designed light patterns (i.e., adversarial projections) onto scenes to deceive deep image classifiers. It has potential applications in privacy protection and the development of more robust classifiers. However, existing approaches primarily focus on individual classifiers and fixed camera poses, often neglecting the complexities of multi-classifier systems and scenarios with varying camera poses. This limitation reduces their effectiveness when introducing new classifiers or camera poses. In this paper, we introduce Classifier-Agnostic Projector-Based Adversarial Attack (CAPAA) to address these issues. First, we develop a novel classifier-agnostic adversarial loss and optimization framework that aggregates adversarial and stealthiness loss gradients from multiple classifiers. Then, we propose an attention-based gradient weighting mechanism that concentrates perturbations on regions of high classification activation, thereby improving the robustness of adversarial projections when applied to scenes with varying camera poses. Our extensive experimental evaluations demonstrate that CAPAA achieves both a higher attack success rate and greater stealthiness compared to existing baselines. Codes are available at: https://github.com/ZhanLiQxQ/CAPAA.

摘要: 基于投影仪的对抗攻击旨在投影精心设计的光图案（即，对抗性投影）到场景上以欺骗深层图像分类器。它在隐私保护和开发更强大的分类器方面具有潜在的应用。然而，现有的方法主要关注单个分类器和固定的相机姿态，通常忽视了多分类器系统和具有不同相机姿态的场景的复杂性。这种限制降低了引入新分类器或相机姿势时的有效性。在本文中，我们引入分类不可知投影仪的对抗攻击（CAPPA）来解决这些问题。首先，我们开发了一个新颖的分类器不可知的对抗性损失和优化框架，该框架聚合了来自多个分类器的对抗性损失和隐蔽性损失梯度。然后，我们提出了一种基于注意力的梯度加权机制，该机制将扰动集中在高分类激活区域，从而提高了应用于摄像机姿态变化的场景时对抗投影的鲁棒性。我们广泛的实验评估表明，CAPAA实现了更高的攻击成功率和更大的隐蔽性相比，现有的基线。代码可访问：https://github.com/ZhanLiQxQ/CAPAA。



## **32. Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability**

用保形预测增强对抗鲁棒性：保证模型可靠性的框架 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07804v1) [paper-pdf](http://arxiv.org/pdf/2506.07804v1)

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction.

摘要: 随着深度学习模型越来越多地部署在高风险应用中，针对对抗攻击的强大防御和可靠的性能保证变得至关重要。此外，仅靠准确性并不能为这些模型提供足够的保证或可靠的不确定性估计。这项研究通过利用保形预测的原则来推进对抗训练。具体来说，我们开发了一种对抗攻击方法，称为OPSA（最佳大小攻击），旨在通过在不要求覆盖保证的情况下最大化模型不确定性来降低任何重要性水平上的保形预测的效率。相应地，我们引入了OPSA-AT（对抗训练），这是一种将OPSA集成到新型适形训练范式中的防御策略。实验评估表明，与各种防御的基线方法相比，我们的OPSA攻击方法会引发更大的不确定性。相反，我们的OPSA-AT防御模型不仅显着增强了针对OPSA以及其他对抗攻击的鲁棒性，并保持了可靠的预测。我们的研究结果凸显了这种集成方法对于为安全关键领域开发值得信赖和有弹性的深度学习模型的有效性。我们的代码可在https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction上获取。



## **33. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

审判与信任：以全面的防御战略应对拜占庭袭击 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.07614v2) [paper-pdf](http://arxiv.org/pdf/2505.07614v2)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.

摘要: 机器学习的最新进展提高了性能，同时也增加了计算需求。虽然联邦和分布式设置可以解决这些问题，但其结构很容易受到恶意影响。在本文中，我们解决了一个特定的威胁，即拜占庭攻击，其中受影响的客户端注入对抗性更新以破坏全球融合。我们将信任分数概念与尝试函数方法相结合，以动态过滤离群值。我们的方法解决了以前方法的关键局限性，即使在拜占庭节点占多数时也允许功能。此外，我们的算法适用于Adam和RMSProp等广泛使用的缩放方法，以及实际场景，包括本地训练和部分参与。我们通过对从医疗机构收集的合成和真实心电图数据进行广泛的实验来验证我们方法的稳健性。此外，我们还对算法及其对上述实际设置的扩展进行了广泛的理论分析。我们方法的收敛保证与没有拜占庭干扰而开发的经典算法的收敛保证相当。



## **34. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。这些风险可能会因最近的对抗性攻击、微调漏洞以及在高风险环境中越来越多地部署LLM而放大。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新的方法，从根本上破坏了LLM中有害行为的表示，提供了一种可扩展的解决方案来增强（潜在的固有）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **35. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



## **36. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

ProARD：渐进式对抗稳健性蒸馏：提供广泛的稳健学生 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07666v1) [paper-pdf](http://arxiv.org/pdf/2506.07666v1)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.

摘要: 对抗鲁棒性蒸馏（ARD）已成为增强轻量级深度神经网络抵御对抗攻击鲁棒性的有效方法。当前的ARD方法利用了一个强大的教师网络来培训一个强大的轻量级学生。然而，由于边缘设备的多样性和资源限制，当前的方法需要从头开始训练新的学生网络以满足特定的限制，从而导致巨大的计算成本和二氧化碳排放量增加。本文提出了渐进对抗鲁棒蒸馏（ProARD），可以对动态网络进行高效的一次性训练，该网络支持各种准确且稳健的学生网络，而无需再培训。我们首先基于动态层构建动态深度神经网络，通过涵盖每个设计阶段的宽度、深度和扩展的变化，以支持广泛的架构。然后，我们将规模最大的学生网络视为动态教师网络。ProARD使用权重共享机制训练这个动态网络，以联合优化动态教师网络及其内部学生网络。然而，由于计算动态网络中所有学生的精确梯度的计算成本很高，因此需要采样机制来选择学生的子集。我们表明，每次迭代中的随机学生抽样无法产生准确和稳健的学生。



## **37. Feature Statistics with Uncertainty Help Adversarial Robustness**

具有不确定性的特征统计有助于对抗稳健性 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2503.20583v2) [paper-pdf](http://arxiv.org/pdf/2503.20583v2)

**Authors**: Ran Wang, Xinlei Zhou, Meng Hu, Rihao Li, Wenhui Wu, Yuheng Jia

**Abstract**: Despite the remarkable success of deep neural networks (DNNs), the security threat of adversarial attacks poses a significant challenge to the reliability of DNNs. In this paper, both theoretically and empirically, we discover a universal phenomenon that has been neglected in previous works, i.e., adversarial attacks tend to shift the distributions of feature statistics. Motivated by this finding, and by leveraging the advantages of uncertainty-aware stochastic methods in building robust models efficiently, we propose an uncertainty-driven feature statistics adjustment module for robustness enhancement, named Feature Statistics with Uncertainty (FSU). It randomly resamples channel-wise feature means and standard deviations of examples from multivariate Gaussian distributions, which helps to reconstruct the perturbed examples and calibrate the shifted distributions. The calibration recovers some domain characteristics of the data for classification, thereby mitigating the influence of perturbations and weakening the ability of attacks to deceive models. The proposed FSU module has universal applicability in training, attacking, predicting, and fine-tuning, demonstrating impressive robustness enhancement ability at a trivial additional time cost. For example, by fine-tuning the well-established models with FSU, the state-of-the-art methods achieve up to 17.13% and 34.82% robustness improvement against powerful AA and CW attacks on benchmark datasets.

摘要: 尽管深度神经网络（DNN）取得了显着的成功，但对抗性攻击的安全威胁对DNN的可靠性构成了重大挑战。本文从理论上和经验上，我们发现了一个在以前的著作中被忽视的普遍现象，即对抗性攻击往往会改变特征统计数据的分布。受这一发现的启发，并利用不确定性感知随机方法在有效构建稳健模型方面的优势，我们提出了一种不确定性驱动的特征统计调整模块，用于增强稳健性，称为具有不确定性的特征统计（FSU）。它从多元高斯分布中随机重新采样示例的通道特征均值和标准差，这有助于重建受扰动的示例并校准移动的分布。校准恢复数据的一些领域特征进行分类，从而减轻扰动的影响并削弱攻击欺骗模型的能力。提出的FSU模块在训练、攻击、预测和微调方面具有普遍适用性，以微不足道的额外时间成本展示了令人印象深刻的鲁棒性增强能力。例如，通过使用FSU对成熟的模型进行微调，最先进的方法可以针对对基准数据集的强大AA和CW攻击实现高达17.13%和34.82%的稳健性改进。



## **38. RAID: A Dataset for Testing the Adversarial Robustness of AI-Generated Image Detectors**

RAGE：用于测试人工智能生成图像检测器对抗鲁棒性的数据集 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.03988v3) [paper-pdf](http://arxiv.org/pdf/2506.03988v3)

**Authors**: Hicham Eddoubi, Jonas Ricker, Federico Cocchi, Lorenzo Baraldi, Angelo Sotgiu, Maura Pintor, Marcella Cornia, Lorenzo Baraldi, Asja Fischer, Rita Cucchiara, Battista Biggio

**Abstract**: AI-generated images have reached a quality level at which humans are incapable of reliably distinguishing them from real images. To counteract the inherent risk of fraud and disinformation, the detection of AI-generated images is a pressing challenge and an active research topic. While many of the presented methods claim to achieve high detection accuracy, they are usually evaluated under idealized conditions. In particular, the adversarial robustness is often neglected, potentially due to a lack of awareness or the substantial effort required to conduct a comprehensive robustness analysis. In this work, we tackle this problem by providing a simpler means to assess the robustness of AI-generated image detectors. We present RAID (Robust evaluation of AI-generated image Detectors), a dataset of 72k diverse and highly transferable adversarial examples. The dataset is created by running attacks against an ensemble of seven state-of-the-art detectors and images generated by four different text-to-image models. Extensive experiments show that our methodology generates adversarial images that transfer with a high success rate to unseen detectors, which can be used to quickly provide an approximate yet still reliable estimate of a detector's adversarial robustness. Our findings indicate that current state-of-the-art AI-generated image detectors can be easily deceived by adversarial examples, highlighting the critical need for the development of more robust methods. We release our dataset at https://huggingface.co/datasets/aimagelab/RAID and evaluation code at https://github.com/pralab/RAID.

摘要: 人工智能生成的图像已经达到了人类无法可靠地将其与真实图像区分开来的质量水平。为了抵消欺诈和虚假信息的固有风险，检测人工智能生成的图像是一项紧迫的挑战和一个活跃的研究课题。虽然提出的许多方法声称可以实现高检测准确性，但它们通常是在理想化条件下进行评估的。特别是，对抗稳健性经常被忽视，这可能是由于缺乏意识或进行全面稳健性分析所需的大量努力。在这项工作中，我们通过提供一种更简单的方法来评估人工智能生成的图像检测器的稳健性来解决这个问题。我们介绍了RAIDs（人工智能生成图像检测器的稳健评估），这是一个包含72 k个多样化且高度可转移的对抗示例的数据集。该数据集是通过对七个最先进的检测器和由四种不同的文本到图像模型生成的图像进行攻击来创建的。大量实验表明，我们的方法可以生成对抗图像，这些图像以很高的成功率传输到未见的检测器，可以用于快速提供对检测器对抗鲁棒性的大致但仍然可靠的估计。我们的研究结果表明，当前最先进的人工智能生成图像检测器很容易被对抗性示例所欺骗，这凸显了开发更稳健方法的迫切需要。我们在https://huggingface.co/datasets/aimagelab/RAID上发布我们的数据集，并在https://github.com/pralab/RAID上发布评估代码。



## **39. Explore the vulnerability of black-box models via diffusion models**

通过扩散模型探索黑匣子模型的脆弱性 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07590v1) [paper-pdf](http://arxiv.org/pdf/2506.07590v1)

**Authors**: Jiacheng Shi, Yanfu Zhang, Huajie Shao, Ashley Gao

**Abstract**: Recent advancements in diffusion models have enabled high-fidelity and photorealistic image generation across diverse applications. However, these models also present security and privacy risks, including copyright violations, sensitive information leakage, and the creation of harmful or offensive content that could be exploited maliciously. In this study, we uncover a novel security threat where an attacker leverages diffusion model APIs to generate synthetic images, which are then used to train a high-performing substitute model. This enables the attacker to execute model extraction and transfer-based adversarial attacks on black-box classification models with minimal queries, without needing access to the original training data. The generated images are sufficiently high-resolution and diverse to train a substitute model whose outputs closely match those of the target model. Across the seven benchmarks, including CIFAR and ImageNet subsets, our method shows an average improvement of 27.37% over state-of-the-art methods while using just 0.01 times of the query budget, achieving a 98.68% success rate in adversarial attacks on the target model.

摘要: 扩散模型的最新进展使各种应用能够生成高保真度和真实感的图像。然而，这些模型也存在安全和隐私风险，包括版权侵犯、敏感信息泄露以及创建可能被恶意利用的有害或攻击性内容。在这项研究中，我们发现了一种新型安全威胁，攻击者利用扩散模型API来生成合成图像，然后使用合成图像来训练高性能的替代模型。这使攻击者能够以最少的查询对黑匣子分类模型执行模型提取和基于传输的对抗攻击，而无需访问原始训练数据。生成的图像具有足够高的分辨率和多样性，可以训练出一个替代模型，其输出与目标模型的输出密切匹配。在包括CIFAR和ImageNet子集在内的七个基准测试中，我们的方法比最先进的方法平均提高了27.37%，同时仅使用了0.01倍的查询预算，在目标模型上实现了98.68%的对抗攻击成功率。



## **40. MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity**

MalGEN：一个用于网络安全恶意软件建模的生成代理框架 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07586v1) [paper-pdf](http://arxiv.org/pdf/2506.07586v1)

**Authors**: Bikash Saha, Sandeep Kumar Shukla

**Abstract**: The dual use nature of Large Language Models (LLMs) presents a growing challenge in cybersecurity. While LLM enhances automation and reasoning for defenders, they also introduce new risks, particularly their potential to be misused for generating evasive, AI crafted malware. Despite this emerging threat, the research community currently lacks controlled and extensible tools that can simulate such behavior for testing and defense preparation. We present MalGEN, a multi agent framework that simulates coordinated adversarial behavior to generate diverse, activity driven malware samples. The agents work collaboratively to emulate attacker workflows, including payload planning, capability selection, and evasion strategies, within a controlled environment built for ethical and defensive research. Using MalGEN, we synthesized ten novel malware samples and evaluated them against leading antivirus and behavioral detection engines. Several samples exhibited stealthy and evasive characteristics that bypassed current defenses, validating MalGEN's ability to model sophisticated and new threats. By transforming the threat of LLM misuse into an opportunity for proactive defense, MalGEN offers a valuable framework for evaluating and strengthening cybersecurity systems. The framework addresses data scarcity, enables rigorous testing, and supports the development of resilient and future ready detection strategies.

摘要: 大型语言模型（LLM）的双重用途性质给网络安全带来了越来越大的挑战。虽然LLM增强了防御者的自动化和推理，但它们也带来了新的风险，特别是它们被滥用来生成规避的、人工智能精心设计的恶意软件的可能性。尽管存在这种新出现的威胁，但研究界目前缺乏可以模拟此类行为以进行测试和防御准备的受控和可扩展的工具。我们介绍了Malgen，这是一个多代理框架，可以模拟协调的对抗行为，以生成多样化的、活动驱动的恶意软件样本。这些代理在为道德和防御研究而构建的受控环境中协作模拟攻击者的工作流程，包括有效负载规划、能力选择和规避策略。使用Malgen，我们合成了十个新型恶意软件样本，并针对领先的防病毒和行为检测引擎对其进行了评估。几个样本表现出绕过当前防御的隐身和规避特征，验证了Malgen建模复杂和新威胁的能力。通过将LLM滥用的威胁转化为积极防御的机会，Malgen为评估和加强网络安全系统提供了一个宝贵的框架。该框架解决了数据稀缺问题，实现了严格的测试，并支持开发有弹性且面向未来的检测策略。



## **41. HSF: Defending against Jailbreak Attacks with Hidden State Filtering**

HSF：利用隐藏状态过滤防御越狱攻击 cs.CR

WWW2025 WSAI BESTPAPER

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2409.03788v2) [paper-pdf](http://arxiv.org/pdf/2409.03788v2)

**Authors**: Cheng Qian, Hainan Zhang, Lei Sha, Zhiming Zheng

**Abstract**: With the growing deployment of LLMs in daily applications like chatbots and content generation, efforts to ensure outputs align with human values and avoid harmful content have intensified. However, increasingly sophisticated jailbreak attacks threaten this alignment, aiming to induce unsafe outputs. Current defense efforts either focus on prompt rewriting or detection, which are limited in effectiveness due to the various design of jailbreak prompts, or on output control and detection, which are computationally expensive as they require LLM inference. Therefore, designing a pre-inference defense method that resists diverse jailbreak prompts is crucial for preventing LLM jailbreak attacks. We observe that jailbreak attacks, safe queries, and harmful queries exhibit different clustering patterns within the LLM's hidden state representation space. This suggests that by leveraging the LLM's hidden state representational capabilities, we can analyze the LLM's forthcoming behavior and proactively intervene for defense. In this paper, we propose a jailbreak attack defense strategy based on a Hidden State Filter (HSF), a lossless architectural defense mechanism that enables the model to preemptively identify and reject adversarial inputs before the inference process begins. We activate its defensive potential through an additional plugin module, effectively framing the defense task as a classification problem. Experimental results on two benchmark datasets, utilizing three different LLMs, show that HSF significantly enhances resilience against six cutting-edge jailbreak attacks. It significantly reduces the success rate of jailbreak attacks while minimally impacting responses to benign user queries, with negligible inference overhead, and outperforming defense baselines.Our code and data are available at https://anonymous.4open.science/r/Hidden-State-Filtering-8652/

摘要: 随着LLM在聊天机器人和内容生成等日常应用程序中的部署越来越多，确保产出与人类价值观保持一致并避免有害内容的努力得到了加强。然而，越来越复杂的越狱攻击威胁着这种一致，旨在引发不安全的输出。当前的防御工作要么集中在提示重写或检测上，由于越狱提示的各种设计，这些工作的有效性受到限制，要么集中在输出控制和检测上，因为它们需要LLM推理，计算成本很高。因此，设计一种抵御不同越狱提示的预推理防御方法对于防止LLM越狱攻击至关重要。我们观察到，越狱攻击，安全的查询，有害的查询表现出不同的聚类模式在LLM的隐藏状态表示空间。这表明，通过利用LLM的隐藏状态表示能力，我们可以分析LLM即将发生的行为，并主动干预防御。在本文中，我们提出了一种基于隐藏状态过滤器（HSF）的越狱攻击防御策略，这是一种无损的架构防御机制，使模型能够在推理过程开始之前抢先识别和拒绝敌对输入。我们通过一个额外的插件模块激活其防御潜力，有效地将防御任务视为一个分类问题。利用三种不同的LLM对两个基准数据集的实验结果表明，HSF显着增强了针对六种尖端越狱攻击的弹性。它显着降低了越狱攻击的成功率，同时对良性用户查询的响应的影响最小，推断费用可以忽略不计，并且优于防御基线。我们的代码和数据可在https://anonymous.4open.science/r/Hidden-State-Filtering-8652/上获取



## **42. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

攻击基础模型的注意力会扰乱下游任务 cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05394v2) [paper-pdf](http://arxiv.org/pdf/2506.05394v2)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention

摘要: 基础模型代表了人工智能领域最突出、最新的范式转变。基础模型是大型模型，基于广泛的数据进行训练，可以在许多下游任务中提供高准确性，通常无需微调。因此，CLIP、DINO或Vision Transfomers（ViT）等型号正在成为许多工业人工智能应用的基石。然而，对预训练的基础模型的依赖也带来了严重的安全问题，因为这些模型容易受到对抗攻击。此类攻击涉及故意设计的输入，旨在欺骗人工智能系统，危及其可靠性。本文研究了视觉基础模型的漏洞，特别关注CLIP和ViT，并探讨了对抗性攻击到下游任务的可转移性。我们引入了一种新颖的攻击，以任务不可知的方式针对基于转换器的架构的结构。我们展示了我们对几个下游任务的攻击的有效性：分类、字幕、图像/文本检索、分割和深度估计。代码可访问：https://github.com/HondamunigePrasannaSilva/attack-attention



## **43. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

通过在线自玩强化学习来追逐移动目标，以实现更安全的语言模型 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07468v1) [paper-pdf](http://arxiv.org/pdf/2506.07468v1)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).

摘要: 传统语言模型（LM）安全对齐依赖于反应性、不相交的过程：攻击者利用静态模型，然后进行防御性微调以修补暴露的漏洞。这种顺序方法造成了不匹配--攻击者过度适应过时的防御，而防御者则永远落后于新兴威胁。为了解决这个问题，我们提出了Self-RedTeam，这是一种在线自玩强化学习算法，攻击者和防御者代理通过持续的交互共同进化。我们将安全调整视为一个两人零和游戏，其中单一模型在攻击者和防御者角色之间交替--生成对抗性提示并防范它们--而奖励LM则判定结果。这实现了动态协同适应。我们以零和游戏的博弈论框架为基础，建立了一个理论安全保证，这激励了我们的方法的设计：如果自我游戏收敛于纳什均衡，防御者将可靠地对任何对抗输入产生安全反应。从经验上看，与针对静态防御者训练的攻击者相比，Self-RedTeam发现了更多样化的攻击（+21.8%SBERT），并在安全基准上实现了更高的稳健性（例如，WildJailBreak上+65.5%）比防守者训练对抗静态攻击者。我们进一步提出隐藏的思想链，允许代理人私下计划，这可以增强对抗多样性并减少过度拒绝。我们的结果促使LM安全培训从反应性修补转向主动协同进化，通过多代理强化学习（MARL）实现LM的可扩展、自主和稳健的自我改进。



## **44. A Red Teaming Roadmap Towards System-Level Safety**

迈向系统级安全的红色团队路线图 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05376v2) [paper-pdf](http://arxiv.org/pdf/2506.05376v2)

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future.

摘要: 实现请求拒绝的大型语言模型（LLM）保障措施已成为一种广泛采用的针对滥用的缓解策略。在对抗性机器学习和人工智能安全的交叉点上，红色防护有效地识别了最先进的再培训LL中的关键漏洞。然而，我们认为，许多关于LLM红色团队的会议提交的文件总体上并没有优先考虑正确的研究问题。首先，针对明确的产品安全规范进行测试应该比抽象的社会偏见或道德原则更优先。其次，红色团队应该优先考虑现实的威胁模型，这些模型代表不断扩大的风险格局以及真正的攻击者可能会做什么。最后，我们认为系统级安全是推进红色团队研究的必要步骤，因为人工智能模型呈现了新的威胁以及威胁缓解的可供性（例如，检测和禁止恶意用户）一旦置于部署上下文中。为了让红色团队研究充分解决人工智能快速发展当今和不久的将来出现的一系列新威胁，采取这些优先事项是必要的。



## **45. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2502.20650v3) [paper-pdf](http://arxiv.org/pdf/2502.20650v3)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获得。



## **46. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

不确定性和校准对可能性比隶属推理攻击的影响 cs.IT

16 pages, 28 figures

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2402.10686v5) [paper-pdf](http://arxiv.org/pdf/2402.10686v5)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在隶属推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否用于训练目标模型。在本文中，我们分析了似然比攻击（LiRA）的性能在一个信息理论框架内，允许调查的影响任意的不确定性在真实的数据生成过程中，由有限的训练数据集造成的认知不确定性，和目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型收到的信息反馈越来越少：置信向量（CV）披露，其中输出概率向量被释放;真实标签置信度（TLC）披露，其中只有分配给真实标签的概率由模型提供;以及决策集（DS）公开，其中如在共形预测中一样产生自适应预测集。我们得出了MIA对手的优势界限，旨在深入了解不确定性和校准对MIA有效性的影响。仿真结果表明，推导出的分析界能够很好地预测MIA的有效性。



## **47. Defending Against Diverse Attacks in Federated Learning Through Consensus-Based Bi-Level Optimization**

通过基于启发的双层优化防御联邦学习中的各种攻击 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2412.02535v2) [paper-pdf](http://arxiv.org/pdf/2412.02535v2)

**Authors**: Nicolás García Trillos, Aditya Kumar Akash, Sixu Li, Konstantin Riedl, Yuhua Zhu

**Abstract**: Adversarial attacks pose significant challenges in many machine learning applications, particularly in the setting of distributed training and federated learning, where malicious agents seek to corrupt the training process with the goal of jeopardizing and compromising the performance and reliability of the final models. In this paper, we address the problem of robust federated learning in the presence of such attacks by formulating the training task as a bi-level optimization problem. We conduct a theoretical analysis of the resilience of consensus-based bi-level optimization (CB$^2$O), an interacting multi-particle metaheuristic optimization method, in adversarial settings. Specifically, we provide a global convergence analysis of CB$^2$O in mean-field law in the presence of malicious agents, demonstrating the robustness of CB$^2$O against a diverse range of attacks. Thereby, we offer insights into how specific hyperparameter choices enable to mitigate adversarial effects. On the practical side, we extend CB$^2$O to the clustered federated learning setting by proposing FedCB$^2$O, a novel interacting multi-particle system, and design a practical algorithm that addresses the demands of real-world applications. Extensive experiments demonstrate the robustness of the FedCB$^2$O algorithm against label-flipping attacks in decentralized clustered federated learning scenarios, showcasing its effectiveness in practical contexts.

摘要: 对抗性攻击对许多机器学习应用程序构成了重大挑战，特别是在分布式训练和联邦学习的环境中，恶意代理试图破坏训练过程，目的是危害和损害最终模型的性能和可靠性。在本文中，我们通过将训练任务描述为双层优化问题来解决存在此类攻击时的鲁棒联邦学习问题。我们对对抗环境下基于共识的双层优化（CB $' 2$O）的弹性进行了理论分析，这是一种交互式多粒子元启发式优化方法。具体来说，我们提供了在存在恶意代理的情况下平均场定律中CB$#2$O的全球收敛分析，证明了CB$#2$O对各种攻击的稳健性。因此，我们深入了解特定的超参数选择如何减轻对抗影响。在实践方面，我们通过提出FedCB $#2$O（一种新型交互多粒子系统）将CB $#2$O扩展到集群联邦学习环境，并设计了一种可满足现实世界应用程序需求的实用算法。大量实验证明了FedCB $' 2$O算法在去中心化集群联邦学习场景中对抗标签翻转攻击的稳健性，展示了其在实际环境中的有效性。



## **48. Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation**

具有隐形语义操纵的视觉语言模型的后门攻击 cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07214v1) [paper-pdf](http://arxiv.org/pdf/2506.07214v1)

**Authors**: Zhiyuan Zhong, Zhen Sun, Yepang Liu, Xinlei He, Guanhong Tao

**Abstract**: Vision Language Models (VLMs) have shown remarkable performance, but are also vulnerable to backdoor attacks whereby the adversary can manipulate the model's outputs through hidden triggers. Prior attacks primarily rely on single-modality triggers, leaving the crucial cross-modal fusion nature of VLMs largely unexplored. Unlike prior work, we identify a novel attack surface that leverages cross-modal semantic mismatches as implicit triggers. Based on this insight, we propose BadSem (Backdoor Attack with Semantic Manipulation), a data poisoning attack that injects stealthy backdoors by deliberately misaligning image-text pairs during training. To perform the attack, we construct SIMBad, a dataset tailored for semantic manipulation involving color and object attributes. Extensive experiments across four widely used VLMs show that BadSem achieves over 98% average ASR, generalizes well to out-of-distribution datasets, and can transfer across poisoning modalities. Our detailed analysis using attention visualization shows that backdoored models focus on semantically sensitive regions under mismatched conditions while maintaining normal behavior on clean inputs. To mitigate the attack, we try two defense strategies based on system prompt and supervised fine-tuning but find that both of them fail to mitigate the semantic backdoor. Our findings highlight the urgent need to address semantic vulnerabilities in VLMs for their safer deployment.

摘要: 视觉语言模型（VLM）已表现出出色的性能，但也容易受到后门攻击，对手可以通过隐藏触发器操纵模型的输出。之前的攻击主要依赖于单模式触发，使得VLM的关键跨模式融合本质基本上没有被探索。与之前的工作不同，我们发现了一种新颖的攻击表面，它利用跨模式语义不匹配作为隐式触发器。基于这一见解，我们提出了BadSem（具有语义操纵的后门攻击），这是一种数据中毒攻击，通过在训练期间故意错位图像-文本对来注入隐形后门。为了执行攻击，我们构建了SIMBad，这是一个专为涉及颜色和对象属性的语义操作而定制的数据集。对四种广泛使用的VLM进行的广泛实验表明，BadSem的平均ASC率超过98%，很好地推广到分布外数据集，并且可以跨中毒模式传输。我们使用注意力可视化进行的详细分析表明，后门模型在不匹配的条件下专注于语义敏感区域，同时在干净的输入上保持正常行为。为了减轻攻击，我们尝试了两种基于系统提示和监督微调的防御策略，但发现这两种策略都未能减轻语义后门。我们的研究结果凸显了迫切需要解决VLM中的语义漏洞，以更安全地部署它们。



## **49. Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models**

质量多样性红色团队化：针对大型语言模型自动生成高质量且多样化的攻击者 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07121v1) [paper-pdf](http://arxiv.org/pdf/2506.07121v1)

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs.

摘要: 确保大型语言模型（LLM）的安全性非常重要。红色团队--一种识别引发目标LLM有害反应的对抗提示的系统方法--已成为一种至关重要的安全评估方法。在此框架下，对抗提示的多样性对于全面的安全评估至关重要。我们发现以前的红色团队方法可能存在两个关键限制。首先，他们经常通过词频或句子嵌入相似度等简单化指标来追求多样性，这可能无法捕捉攻击策略中有意义的变化。其次，训练单一攻击者模型的常见做法限制了潜在攻击风格和风险类别的覆盖范围。本文介绍了质量多样性红色团队（QDRT），这是一个旨在解决这些限制的新框架。QDRT通过行为条件训练实现目标驱动的多样性，并以开放式方式实现行为回放缓冲区。此外，它还培训了多个专业攻击者，能够在不同的风格和风险类别中生成高质量的攻击。我们的经验评估表明，QDRT生成的攻击更多样化，对各种目标LLM更有效，包括GPT-2，Llama-3，Gemma-2和Qwen2.5。这项工作通过提供一种系统有效的方法来自动化红队，最终支持LLM的负责任部署，从而推进了LLM安全领域。



## **50. D2R: dual regularization loss with collaborative adversarial generation for model robustness**

D2R：双正则化损失与协作对抗生成模型鲁棒性 cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07056v1) [paper-pdf](http://arxiv.org/pdf/2506.07056v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Rajiv Ranjan, Zhanxing Zhu, Vaclav Snasel, Varun Ojha

**Abstract**: The robustness of Deep Neural Network models is crucial for defending models against adversarial attacks. Recent defense methods have employed collaborative learning frameworks to enhance model robustness. Two key limitations of existing methods are (i) insufficient guidance of the target model via loss functions and (ii) non-collaborative adversarial generation. We, therefore, propose a dual regularization loss (D2R Loss) method and a collaborative adversarial generation (CAG) strategy for adversarial training. D2R loss includes two optimization steps. The adversarial distribution and clean distribution optimizations enhance the target model's robustness by leveraging the strengths of different loss functions obtained via a suitable function space exploration to focus more precisely on the target model's distribution. CAG generates adversarial samples using a gradient-based collaboration between guidance and target models. We conducted extensive experiments on three benchmark databases, including CIFAR-10, CIFAR-100, Tiny ImageNet, and two popular target models, WideResNet34-10 and PreActResNet18. Our results show that D2R loss with CAG produces highly robust models.

摘要: 深度神经网络模型的稳健性对于保护模型免受对抗攻击至关重要。最近的防御方法采用协作学习框架来增强模型的稳健性。现有方法的两个关键局限性是（i）通过损失函数对目标模型的指导不足;（ii）非协作对抗生成。因此，我们提出了一种双重正规化损失（D2 R损失）方法和一种用于对抗训练的协作对抗生成（COG）策略。D2 R损失包括两个优化步骤。对抗性分布和干净分布优化通过利用通过适当的函数空间探索获得的不同损失函数的强度来更精确地关注目标模型的分布，增强了目标模型的鲁棒性。MAG使用引导模型和目标模型之间基于梯度的协作来生成对抗样本。我们对三个基准数据库进行了广泛的实验，包括CIFAR-10、CIFAR-100、Tiny ImageNet以及两个流行的目标模型WideResNet 34 -10和PreActResNet 18。我们的结果表明，随着MAG的D2 R损失会产生高度稳健的模型。



