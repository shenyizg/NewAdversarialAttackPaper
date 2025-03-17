# Latest Large Language Model Attack Papers
**update at 2025-03-17 10:32:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

针锋相对：通过对抗性防御保护大型视觉语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.

摘要: 部署大型视觉语言模型(LVLM)引入了一个独特的漏洞：通过视觉输入易受恶意攻击。然而，现有的防御方法有两个关键的局限性：(1)它们只关注文本防御，不能直接应对发起攻击的视域中的威胁；(2)额外的处理步骤往往会导致巨大的计算开销或对良性任务的模型性能造成影响。基于这些见解，我们提出了ESIII(Embedding Security Instructions To Images)，这是一种将视觉空间从易受攻击源转变为主动防御机制的新方法。首先，通过基于梯度的优化将安全指令嵌入到防御图像中，得到视觉维度的安全指令。随后，我们将来自视觉和文本维度的安全指令与输入查询集成在一起。来自不同维度的安全指令之间的协作确保了全面的安全防护。大量的实验表明，我们的方法有效地增强了LVLM对此类攻击的健壮性，同时保持了它们在标准良性任务上的性能，并导致了不可察觉的时间开销的增加。



## **2. Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification**

深度结盟：通过渐进式答案去规范化捍卫越狱袭击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11185v1) [paper-pdf](http://arxiv.org/pdf/2503.11185v1)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击使用精心编制的提示来引发有毒响应。这些攻击利用了LLMS在生成过程中动态检测有害意图的困难。传统的安全配准方法往往依赖于初始的几个生成步骤，由于计算预算有限，效率不高。本文提出了DEEPALIGN，这是一个健壮的防御框架，它对LLMS进行微调，以逐步对生成的内容进行解毒，显著提高了计算预算和减轻有害生成的有效性。我们的方法使用一种对隐态进行操作的混合损失函数来直接提高LLMS在生成过程中对毒性的固有意识。此外，我们通过生成对有害查询的语义相关答案来重新定义安全响应，从而提高了对表示-突变攻击的健壮性。对多个LLM的评估显示了针对六种不同攻击类型的最先进的防御性能，与以前最先进的防御相比，攻击成功率降低了高达两个数量级，同时保留了实用性。这项工作通过动态、上下文感知的缓解来解决传统对准的局限性，从而提高了LLM的安全性。



## **3. Towards a Systematic Evaluation of Hallucinations in Large-Vision Language Models**

对大视野语言模型中的幻觉进行系统评估 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.20622v2) [paper-pdf](http://arxiv.org/pdf/2412.20622v2)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in complex multimodal tasks. However, these models still suffer from hallucinations, particularly when required to implicitly recognize or infer diverse visual entities from images for complex vision-language tasks. To address this challenge, we propose HALLUCINOGEN, a novel visual question answering (VQA) benchmark that employs contextual reasoning prompts as hallucination attacks to evaluate the extent of hallucination in state-of-the-art LVLMs. Our benchmark provides a comprehensive study of the implicit reasoning capabilities of these models by first categorizing visual entities based on the ease of recognition in an image as either salient (prominent, visibly recognizable objects such as a car) or latent entities (such as identifying a disease from a chest X-ray), which are not readily visible and require domain knowledge or contextual reasoning for accurate inference. Next, we design hallucination attacks for both types of entities to assess hallucinations in LVLMs while performing various vision-language tasks, such as locating or reasoning about specific entities within an image, where models must perform implicit reasoning by verifying the existence of the queried entity within the image before generating responses. Finally, our extensive evaluations of eleven LVLMs, including powerful open-source models (like LLaMA-3.2 and DeepSeek-V2), commercial models like Gemini, and two hallucination mitigation strategies across multiple datasets, demonstrate that current LVLMs remain susceptible to hallucination attacks.

摘要: 大型视觉语言模型在复杂的多通道任务中表现出显著的性能。然而，这些模型仍然存在幻觉，特别是在复杂的视觉语言任务中需要从图像中隐含地识别或推断不同的视觉实体时。为了应对这一挑战，我们提出了幻觉剂，这是一种新颖的视觉问答基准，它使用上下文推理提示作为幻觉攻击来评估最先进的LVLM中的幻觉程度。我们的基准提供了对这些模型的隐含推理能力的全面研究，首先根据图像中识别的容易程度将视觉实体分类为突出实体(突出的、可视识别的对象，如汽车)或潜在实体(如从胸部X光识别疾病)，这些实体不容易看到，需要领域知识或上下文推理才能进行准确的推理。接下来，我们为两种类型的实体设计幻觉攻击，以评估LVLMS中的幻觉，同时执行各种视觉语言任务，如定位或关于图像中的特定实体进行推理，其中模型必须在生成响应之前通过验证被查询实体在图像中的存在来执行隐式推理。最后，我们对11个LVLM进行了广泛的评估，包括强大的开源模型(如Llama-3.2和DeepSeek-V2)、商业模型(如Gemini)，以及跨多个数据集的两种幻觉缓解策略，表明当前的LVLM仍然容易受到幻觉攻击。



## **4. ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models**

ChatGPT遭遇变形攻击检测：具有多模式大语言模型和通用视觉模型的零镜头MAD cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10937v1) [paper-pdf](http://arxiv.org/pdf/2503.10937v1)

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications.

摘要: 人脸识别系统(FRS)越来越容易受到变形攻击，这促使变形攻击检测(MAD)算法的发展。然而，MAD的一个关键挑战在于，它对看不见的数据的概括性有限，而且缺乏可解释性--这对于招生站和自动边界控制系统等实际应用环境至关重要。认识到大多数现有的MAD算法依赖于有监督的学习范例，该工作探索了一种利用大型语言模型(LLMS)的零镜头学习来实现MAD的新方法。我们提出了两种类型的零镜头MAD算法：一种利用一般的视觉模型，另一种利用多模式LLMS。对于一般的视觉模型，我们通过计算独立支持集的平均支持嵌入来解决MAD任务，而不使用变形图像。对于基于LLM的方法，我们使用了最先进的GPT-4 Turbo API和精心制作的提示。为了评估零镜头MAD的可行性和提出的方法的有效性，我们构建了一个包含各种不可见变形算法的打印扫描变形数据集，模拟了具有挑战性的真实应用场景。实验结果显示了显著的检测精度，验证了零镜头学习在MAD任务中的适用性。此外，我们对基于LLM的MAD的研究表明，多通道LLM，如ChatGPT，对未经训练的MAD任务表现出显著的泛化能力。此外，它们还具有提供解释和指导的独特能力，这可以在实际应用中提高最终用户的透明度和可用性。



## **5. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.

摘要: 尽管在开源的大型视觉语言模型(LVLM)上性能很好，但基于传输的定向攻击对黑盒商业LVLM往往失败。分析失败的对抗性扰动发现，学习的扰动通常源于均匀分布，缺乏明确的语义细节，导致意外响应。这种严重的语义信息缺失导致商业LVLM要么完全忽略该扰动，要么曲解其嵌入的语义，从而导致攻击失败。为了克服这些问题，我们注意到，识别核心语义对象是用各种数据集和方法训练的模型的关键目标。这种洞察力激发了我们的方法，通过在局部区域内编码显式的语义细节来细化语义清晰度，从而确保互操作性和捕获更细粒度的特征，并通过将修改集中在语义丰富的区域而不是统一地应用它们。为了实现这一点，我们提出了一种简单而高效的解决方案：在每个优化步骤中，通过控制纵横比和比例来随机裁剪敌对图像，调整大小，然后在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们的对抗性例子使用聚焦于关键区域的局部聚集扰动来制作，表现出出奇地好的可转换性，可以用于商业LVLM，包括GPT-4.5、GPT-40、Gemini-2.0-Flash、Claude-3.5-十四行诗、Claude-3.7-十四行诗，甚至像o1、Claude-3.7-Think和Gemini-2.0-Flash-Think这样的推理模型。我们的方法在GPT-4.5、40o和o1上的成功率超过90%，大大超过了之前所有最先进的攻击方法。我们在不同配置和训练代码下的优化对抗性示例可在https://github.com/VILA-Lab/M-Attack.获得



## **6. ASIDE: Architectural Separation of Instructions and Data in Language Models**

ASIDE：语言模型中指令和数据的架构分离 cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10566v1) [paper-pdf](http://arxiv.org/pdf/2503.10566v1)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Soroush Tabesh, Alexandra Volkova, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose an architectural change, ASIDE, that allows the model to clearly separate between instructions and data by using separate embeddings for them. Instead of training the embeddings from scratch, we propose a method to convert an existing model to ASIDE form by using two copies of the original model's embeddings layer, and applying an orthogonal rotation to one of them. We demonstrate the effectiveness of our method by showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.

摘要: 尽管性能卓越，但大型语言模型缺乏基本的安全功能，这使得它们容易受到大量恶意攻击。特别是，以前的工作已经确定指令和数据之间没有内在的分离是快速注入攻击成功的根本原因。在这项工作中，我们提出了一种架构改变，允许模型通过对指令和数据使用单独的嵌入来明确地分离它们。我们没有从头开始训练嵌入，而是提出了一种方法，通过使用原始模型的嵌入层的两个副本并对其中一个模型应用正交旋转来将现有模型转换为旁注形式。我们通过展示(1)在不损失模型能力的情况下极大地提高指令-数据分离分数和(2)即使在没有专门的安全培训的情况下也在快速注入基准上具有竞争力的结果来证明我们方法的有效性。此外，我们还通过对模型表示的分析，研究了该方法背后的工作机制。



## **7. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

TH-Bench：通过机器生成文本检测器上人性化人工智能文本来评估逃避攻击 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.

摘要: 随着大型语言模型(LLM)的发展，机器生成文本(MGTS)变得越来越流畅、高质量和信息丰富。现有的大范围MGT探测器旨在识别MGT，以防止抄袭和错误信息的传播。然而，攻击者试图使MGTS人性化以躲避检测(称为躲避攻击)，这只需要进行少量修改即可绕过MGT检测器。遗憾的是，现有的攻击通常缺乏统一和全面的评估框架，因为它们是使用不同的实验设置、模型架构和数据集进行评估的。为了填补这一空白，我们引入了文本人性化基准(TH-BENCH)，这是第一个评估针对MGT检测器的躲避攻击的综合基准。TH-BENCH从三个关键维度对攻击进行评估：规避效率、文本质量和计算开销。我们的广泛实验评估了6种针对6个数据集的13个MGT检测器的最先进攻击，这些攻击跨越19个域，由11个广泛使用的LLM生成。我们的发现表明，没有一次逃避攻击在所有三个维度上都表现出色。通过深入的分析，我们突出了不同攻击的优势和局限性。更重要的是，我们确定了三个维度之间的权衡，并提出了两个优化见解。通过初步实验，验证了它们的正确性和有效性，为以后的研究提供了潜在的方向。



## **8. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

针对大型语言模型协作推理的提示倒置攻击 cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09022v2) [paper-pdf](http://arxiv.org/pdf/2503.09022v2)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.

摘要: 大语言模型以其卓越的内容生成能力得到了广泛的应用。然而，开源LLMS的实际使用受到高资源要求的阻碍，使得部署成本高昂，并限制了广泛的发展。协作推理是解决这一问题的一种很有前途的解决方案，其中每个用户通过托管一个层的子集并传递中间激活来进行协作。许多公司正在构建协作推理平台，以利用用户未充分利用的GPU来降低LLM服务成本。尽管学术界和工业界对协同推理产生了广泛的兴趣，但与LLM协同推理相关的隐私风险尚未得到很好的研究。这在很大程度上是因为LLM激活的反转带来了挑战，因为它具有很强的非线性。为了验证LLM协同推理中隐私威胁的严重性，我们引入了即时反转攻击(PIA)的概念，恶意参与者试图通过其先前参与者发送的激活来恢复输入提示。广泛的实验表明，我们的PIA方法的性能大大优于现有的基线。例如，我们的方法在使用Llama-65B模型的Skytrax数据集上，在反演最大变压器层数时达到了88.4%的令牌精度，而最佳基线方法只达到了22.8%的精度。实验结果验证了PIA攻击的有效性，并突出了其对LLM协同推理系统的实际威胁。



## **9. ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content**

ExtremeAIGC：LMM漏洞针对人工智能生成的极端主义内容进行基准测试 cs.CR

Preprint

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09964v1) [paper-pdf](http://arxiv.org/pdf/2503.09964v1)

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies.

摘要: 大型多模式模型(LMM)越来越容易受到人工智能生成的极端主义内容的影响，包括照片级真实感图像和文本，这些内容可用于绕过安全机制并产生有害输出。然而，现有的用于评估LMM稳健性的数据集提供了对极端主义内容的有限探索，往往缺乏人工智能生成的图像、多样化的图像生成模型以及对历史事件的全面覆盖，这阻碍了对模型漏洞的完整评估。为了填补这一空白，我们引入了ExtremeAIGC，这是一个基准数据集和评估框架，旨在评估针对此类内容的LMM漏洞。ExtremeAIGC通过策划使用最先进的图像生成技术制作的各种基于文本和图像的示例来模拟真实世界的事件和恶意使用案例。我们的研究揭示了LMM令人震惊的弱点，表明即使是尖端的安全措施也无法防止极端主义材料的产生。我们系统地量化了各种攻击策略的成功率，暴露了当前防御中的关键差距，并强调了需要更强大的缓解策略。



## **10. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.

摘要: 大型语言模型(LLM)最近在时间序列预测方面显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验包括LLMTime与GPT-3.5、GPT-4、Llama和Mistral、TimeGPT和TimeLLM的模型，实验表明，对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。代码存储库可在https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.上找到



## **11. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.

摘要: 将大型语言模型(LLM)集成到网络安全应用程序中提供了重大机遇，如增强威胁分析和恶意软件检测，但也可能带来重大风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMInstruct，这是一个包含54,928个指令-响应对的数据集，涵盖了网络安全任务，如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这包括从多个来源获取数据，将其过滤和组织成指令-响应对，并使其与现实世界的情景相一致，以增强其适用性。选择了7个开源LLM来测试CyberLLMInstruct的有用性：Phi 3 Mini 3.8B、Mistral 7B、Qwen 2.5 7B、Llama 3 8B、Llama 3.1 8B、Gema 2 9B和Llama 2 70B。在我们的主要示例中，我们使用OWASP TOP 10框架严格评估微调模型的安全性，发现微调降低了所有测试的LLM和每个对手攻击的安全弹性(例如，针对即时注入的Llama 3.1 8B的安全分数从0.95下降到0.15)。在我们的第二个示例中，我们展示了这些相同的微调模型在CyberMetric基准上也可以达到高达92.50%的准确率。这些发现突出了性能和安全之间的权衡，显示了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍然提高不同数据集和域的性能。复制数据集所需的所有脚本，以及复制我们的结果的例子和相关资源，将在论文被接受时提供。



## **12. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

对分布式大型语言模型推理框架的提示推理攻击 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09291v1) [paper-pdf](http://arxiv.org/pdf/2503.09291v1)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.

摘要: 现代大型语言模型(LLM)的推理过程需要令人望而却步的计算资源，这使得它们不适合在消费级设备上部署。为了解决这一局限性，最近的研究提出了分布式LLM推理框架，该框架利用分裂学习原理在资源受限的硬件上实现协作式LLM推理。然而，在参与者之间分发LLM层需要传输中间输出，这可能会给原始输入提示带来隐私风险-这是一个尚未在文献中彻底探讨的关键问题。本文通过设计和评估三种旨在从中间LLM输出重构输入提示的提示推理攻击，严格检查了分布式LLM推理框架的隐私漏洞。这些攻击是在各种查询和数据约束下开发的，以反映现实世界中不同的LLM服务场景。具体地说，第一次攻击假定无限制的查询预算和对与目标提示共享相同分布的辅助数据集的访问。第二个攻击也利用无限查询，但使用的辅助数据集具有与目标提示不同的分布。第三种攻击是在限制最严格的情况下进行的，查询预算有限，并且没有可用的辅助数据集。我们在一系列LLM上评估了这些攻击，包括最先进的模型，如Llama-3.2和Phi-3.5，以及广泛使用的模型，如GPT-2和BERT，以进行比较分析。我们的实验表明，前两种攻击的重建精度超过90%，而第三种攻击的重建精度通常在50%以上，即使在严格的约束下也是如此。这些发现突出了分布式LLM推理框架中的隐私风险，对它们在现实世界应用程序中的部署发出了强烈的警告。



## **13. In-Context Defense in Computer Agents: An Empirical Study**

计算机代理中的上下文防御：实证研究 cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.

摘要: 由视觉语言模型(VLM)驱动的计算机代理极大地促进了人机交互，使用户能够通过自然语言指令执行复杂的任务。然而，这些代理容易受到上下文欺骗攻击，这是一种新兴的威胁，即对手在代理的操作环境中嵌入误导性内容，例如包含欺骗性指令的弹出窗口。现有的防御措施，如指示特工忽略欺骗性因素，已被证明在很大程度上无效。作为第一个关于保护计算机代理的系统研究，我们引入了Textbf[上下文中的防御}，利用上下文中的学习和思想链(COT)推理来对抗此类攻击。我们的方法涉及用一小组精心挑选的样本来增强代理的上下文，这些样本既包含恶意环境，也包含相应的防御响应。这些样本指导代理在行动计划之前首先执行明确的防御推理，降低对欺骗性攻击的敏感性。实验证明了该方法的有效性，对弹出窗口攻击的攻击成功率降低了91.2%，对环境注入攻击的平均成功率降低了74.6%，同时对分散注意力的广告实现了100%的成功防御。我们的发现强调：(1)为了获得最佳性能，防御推理必须先于行动计划，(2)最少的样本数量(少于3个)足以诱导代理人的防御行为。



## **14. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **15. A Survey on Trustworthy LLM Agents: Threats and Countermeasures**

值得信赖的LLM代理人调查：威胁与对策 cs.MA

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09648v1) [paper-pdf](http://arxiv.org/pdf/2503.09648v1)

**Authors**: Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, Qingsong Wen

**Abstract**: With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.

摘要: 随着大型语言模型的快速发展，基于大型语言模型的智能体和多智能体系统极大地扩展了大型语言模型生态系统的能力。这种发展源于向LLM提供额外的模块，如内存、工具、环境，甚至其他代理。然而，这一进步也引入了更复杂的可信度问题，以前仅关注LLMS的研究无法涵盖这些问题。在本次调查中，我们提出了TrustAgent框架，这是一项关于代理可信性的综合研究，具有模块化分类、多维内涵和技术实现的特点。通过深入研究和总结新出现的针对代理和多代理系统的攻击、防御和评估方法，我们将可信LLM的概念扩展到新兴的可信代理范式。在TrustAgent中，我们从解构和引入代理和MAS的各种组件开始。然后，我们将他们的可信性分为内在(大脑、记忆和工具)和外在(用户、代理和环境)两个方面。随后，我们描述了可信性的多方面含义，并详细阐述了与这些内部和外部模块相关的现有研究的实现技术。最后，我们提出了我们对这一领域的见解和展望，旨在为未来的努力提供指导。



## **16. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但它们仍然容易受到对手操纵，例如通过即时注入攻击越狱。这些攻击绕过安全机制，生成受限或有害的内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全状态和越狱状态的潜在潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM的激活稳定在半稳定状态，可以被识别和扰动以诱导状态转变。使用降维技术，我们从安全和越狱反应中投影激活，以揭示低维空间中的潜在子空间。然后我们推导出一个扰动向量，当应用于安全表示时，将模型转变为越狱状态。我们的结果表明，这种因果干预导致在提示的子集中有统计意义的越狱反应。接下来，我们探索这些扰动是如何通过模型的各层传播的，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的发现表明，有针对性的扰动导致激活和模型反应的明显转变。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转变为先发制人的、模型不可知的技术，在表征水平上中和敌对状态。



## **17. Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models**

与错误信息作斗争：开源大型语言模型中对抗事实的实证研究 cs.CL

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.10690v1) [paper-pdf](http://arxiv.org/pdf/2503.10690v1)

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information.

摘要: 对抗性真实性是指敌手故意在输入提示中插入错误信息，其特征是表现出不同程度的自信。在这项研究中，我们系统地评估了几个开放源码的大型语言模型(LLM)在面对这种敌意输入时的性能。我们考虑了三个等级的对手自信：强烈自信、适度自信和有限自信。我们的分析包括8个LLMS：骆驼3.1(8B)、Phi 3(3.8B)、Qwen 2.5(7B)、DeepSeek-v2(16B)、Gemma2(9B)、Falcon(7B)、Mstrallite(7B)和LLaVA(7B)。实验结果表明，Llama3.1(8B)表现出较强的检测敌意输入的能力，而Falcon(7B)表现出相对较低的性能。值得注意的是，对于大多数模型，检测成功率随着对手信心的降低而提高；然而，对于大羊驼3.1(8B)和Phi 3(3.8B)，这一趋势相反，其中对手信心的降低与检测性能的降低相对应。对导致最高和最低成功率的查询的进一步分析表明，对抗性攻击在针对不太常被引用或晦涩难懂的信息时更有效。



## **18. JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing**

JBFuzz：使用Fuzzing高效、有效地越狱LLM cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08990v1) [paper-pdf](http://arxiv.org/pdf/2503.08990v1)

**Authors**: Vasudev Gohil

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.   JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.

摘要: 大型语言模型已经显示出作为语言理解和决策工具的巨大前景，它们已经渗透到我们日常生活的各个方面。然而，它们的广泛使用也伴随着新的风险，例如通过一种名为越狱的攻击产生有害的、不道德的或攻击性内容。尽管LLM开发人员做出了广泛的努力，利用人类的反馈来调整LLM，但它们仍然容易受到越狱攻击。为了解决这个问题，研究人员经常使用红色团队来理解和调查越狱提示。然而，现有的红团队方法缺乏有效性、可伸缩性或两者兼而有之。为了解决这些问题，我们提出了JBFuzz，一种新的高效、自动化和可扩展的越狱红色团队技术。JBFuzz的灵感来自于Fuzing在检测软件错误/漏洞方面的成功。我们通过设计新的种子提示、轻量级的突变引擎和轻量级且准确的赋值器来指导模糊器，从而克服了与有效性和可伸缩性相关的三个挑战。吸收所有这三种解决方案的结果是一个强大的模糊，只需要黑匣子访问目标LLM。我们使用九个流行和广泛使用的LLM对JBFuzz进行了广泛的实验评估。我们发现，JBFuzz成功地越狱了所有LLM，涉及各种有害/不道德的问题，平均攻击成功率为99%。我们还发现JBFuzz是非常高效的，因为它平均在60秒内为给定的问题越狱。我们的工作突出了最先进的LLM即使在安全调整之后也容易受到越狱攻击，并且对于LLM开发人员来说是一个有价值的红团队工具。



## **19. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它对闭源模型表现出很强的攻击可转移性，尽管只在Llama3上进行了优化，但在GPT-3.5上达到了99 ASR，在GPT-4上达到了49 ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



## **20. Backtracking for Safety**

为了安全而回溯 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但确保它们的安全性和与人类价值观的一致性仍然至关重要。当前的安全对齐方法，如监督微调和基于强化学习的方法，可能会表现出对对手攻击的脆弱性，并且往往导致浅层安全对齐，主要侧重于防止生成输出的初始令牌中的有害内容。虽然像重置这样的方法可以通过丢弃之前的令牌并重新启动生成过程来帮助从不安全的世代中恢复，但它们不太适合于解决细微差别的安全违规行为，如毒性，否则可能会在良性和漫长的世代中出现。在本文中，我们提出了一种新的回溯方法，旨在解决这些局限性。我们的方法允许模型在发电过程中发生安全违规时恢复到更安全的发电状态，而不一定是在开始时。这种方法能够在不丢弃整个生成的文本的情况下有针对性地纠正有问题的片段，从而保持效率。我们证明，我们的方法大大减少了在产生过程中出现的毒性，对效率的影响最小。



## **21. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

人类可读的对抗性预言：使用情境背景对LLM漏洞的调查 cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.

摘要: 之前发现大型语言模型（LLM）漏洞的研究经常使用无意义的对抗提示。然而，现在可以使用自动检测技术轻松识别此类提示。为了进一步加强对抗性攻击，我们重点关注人类可读的对抗性提示，这些提示是更现实、更强大的威胁。我们的主要贡献是（1）情境驱动攻击利用电影剧本作为上下文来创建人类可读的提示，成功欺骗LLM，（2）对抗性后缀转换，将无意义的对抗性后缀转换为独立有意义的文本，以及（3）具有p核采样的Advancer，一种生成多样化、人类可读的对抗性后缀的方法，提高GPT-3.5和Gemma 7 B等模型的攻击功效。



## **22. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

大型语言模型的主动隐私咨询：保护PRI，对模型实用性的影响可忽略不计 cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.17591v2) [paper-pdf](http://arxiv.org/pdf/2502.17591v2)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.

摘要: 随着大型语言模型(LLM)的兴起，越来越多的研究已经认识到它们在恶意攻击下泄露个人身份信息(PII)的风险。虽然已经做出了努力来保护LLMS中的PII，但现有的方法难以平衡隐私保护和维护模型效用。受认知科学中关于健忘症的研究启发，我们提出了一种新的方法，即主动隐私健忘症(PPA)，在保护LLMS中的PII的同时保持其实用性。这种机制的工作原理是主动识别和忘记序列中与PII最密切相关的关键记忆，然后使用适当的替代记忆植入记忆以维持LLM的功能。我们在多个模型上进行评估，以保护常见的PII，如电话号码和物理地址，免受流行的PII目标攻击，展示了我们的方法与其他现有防御技术相比的优越性。结果表明，我们的PPA方法完全消除了100%的电话号码暴露风险，并显著降低了9.8%-87.6%的物理地址暴露风险，所有这些都保持了可比的模型实用性能。



## **23. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **24. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

对话注入攻击：通过上下文操纵越狱LLM cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.

摘要: 大型语言模型(LLM)在广泛的应用程序中显示出了重要的实用价值；然而，它们的部署受到安全漏洞的困扰，特别是越狱攻击。这些攻击通过精心编制敌意提示来操纵LLM生成有害或不道德的内容。虽然目前对越狱攻击的大部分研究都集中在单回合互动上，但在很大程度上忽视了历史对话对模型行为的影响。在本文中，我们介绍了一种新的越狱范例，对话注入攻击(DIA)，它利用对话历史来提高此类攻击的成功率。DIA在黑盒设置中运行，只需要访问聊天API或了解LLM的聊天模板。我们提出了两种构造对抗性历史对话的方法：一种是采用灰盒预填充攻击，另一种是利用延迟响应。我们的实验表明，DIA在包括Llama-3.1和GPT-40在内的最近的LLM上达到了最先进的攻击成功率。此外，我们还证明了DIA可以绕过5种不同的防御机制，突出了其健壮性和有效性。



## **25. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

针对大型语言模型的多回合越狱攻击的推理增强对话 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.11054v4) [paper-pdf](http://arxiv.org/pdf/2502.11054v4)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.

摘要: 多轮越狱攻击通过在迭代对话中使用大型语言模型(LLM)来模拟真实世界的人类交互，暴露出关键的安全漏洞。然而，现有的方法往往难以在语义一致性和攻击有效性之间取得平衡，导致良性的语义漂移或无效的检测规避。为了应对这一挑战，我们提出了一种新的多轮越狱框架--推理增强对话，该框架将有害的查询重新定义为良性的推理任务，并利用LLMS强大的推理能力来妥协安全对齐。具体地说，我们引入了攻击状态机框架来系统地建模问题转换和迭代推理，确保跨多轮的连贯查询生成。在这个框架的基础上，我们设计了增益引导的探索、自我发挥和拒绝反馈模块，以保留攻击语义，提高有效性，并支持推理驱动的攻击进展。在多个LLM上的大量实验表明，RACE在复杂的会话场景中获得了最先进的攻击效率，攻击成功率(ASR)提高了96%。值得注意的是，我们的方法在领先的商业模型OpenAI o1和DeepSeek r1上分别获得了82%和92%的ASR，这突显了它的有效性。我们在https://github.com/NY1024/RACE上发布我们的代码，以促进在这一关键领域的进一步研究。



## **26. Safety Guardrails for LLM-Enabled Robots**

LLM支持机器人的安全护栏 cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.

摘要: 尽管将大型语言模型(LLM)集成到机器人学中释放了变革性的能力，但它也带来了重大的安全问题，从平均情况下的LLM错误(例如，幻觉)到对抗性的越狱攻击，这可能会在现实世界中产生有害的机器人行为。传统的机器人安全方法不能解决LLM的新脆弱性，而当前的LLM安全护栏忽略了机器人在动态真实环境中操作所带来的物理风险。在本文中，我们提出了一种两级护栏结构RoboGuard，以确保LLM支持的机器人的安全。RoboGuard首先通过使用信任根LLM将预定义的安全规则与机器人环境相关联，该LLM使用思想链(COT)推理来生成严格的安全规范，如时间逻辑约束。然后，RoboGuard使用时态逻辑控制合成来解决这些上下文安全规范和可能不安全的计划之间的潜在冲突，这确保了安全合规性，同时将违反用户偏好的程度降至最低。通过考虑最坏情况下的越狱攻击的大量模拟和真实世界实验，我们证明了RoboGuard在不影响安全计划的性能的情况下，将不安全计划的执行率从92%降低到2.5%以下。我们还证明了RoboGuard是资源高效的，对自适应攻击具有健壮性，并且通过使其信任根LLM能够执行CoT推理而显著增强。这些结果突显了RoboGuard在缓解安全风险和提高启用LLM的机器人可靠性方面的潜力。



## **27. PoisonedParrot: Subtle Data Poisoning Attacks to Elicit Copyright-Infringing Content from Large Language Models**

PoisonedParrot：针对从大型语言模型中获取侵犯版权内容的微妙数据中毒攻击 cs.LG

18 pages, 18 figures. Accepted at NAACL 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07697v1) [paper-pdf](http://arxiv.org/pdf/2503.07697v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Pankayaraj Pathmanathan, Yigitcan Kaya, Zora Che, Bang An, Sicheng Zhu, Aakriti Agrawal, Furong Huang

**Abstract**: As the capabilities of large language models (LLMs) continue to expand, their usage has become increasingly prevalent. However, as reflected in numerous ongoing lawsuits regarding LLM-generated content, addressing copyright infringement remains a significant challenge. In this paper, we introduce PoisonedParrot: the first stealthy data poisoning attack that induces an LLM to generate copyrighted content even when the model has not been directly trained on the specific copyrighted material. PoisonedParrot integrates small fragments of copyrighted text into the poison samples using an off-the-shelf LLM. Despite its simplicity, evaluated in a wide range of experiments, PoisonedParrot is surprisingly effective at priming the model to generate copyrighted content with no discernible side effects. Moreover, we discover that existing defenses are largely ineffective against our attack. Finally, we make the first attempt at mitigating copyright-infringement poisoning attacks by proposing a defense: ParrotTrap. We encourage the community to explore this emerging threat model further.

摘要: 随着大型语言模型(LLM)的功能不断扩展，它们的使用也变得越来越普遍。然而，正如许多正在进行的关于LLM生成的内容的诉讼所反映的那样，解决侵犯版权的问题仍然是一个巨大的挑战。在本文中，我们介绍了PoisonedParot：第一种隐蔽的数据中毒攻击，即使模型没有针对特定的受版权保护的材料进行直接训练，也会诱导LLM生成受版权保护的内容。PoisonedParot使用现成的LLM将受版权保护的文本的小片段集成到毒药样本中。尽管它很简单，在广泛的实验中进行了评估，但PoisonedParot在启动模型以生成受版权保护的内容方面出人意料地有效，没有明显的副作用。此外，我们发现现有的防御在很大程度上对我们的攻击无效。最后，我们首次尝试通过提出一种防御方法来减轻版权侵权中毒攻击：ParrotTrap。我们鼓励社区进一步探索这种新兴的威胁模式。



## **28. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们在相对平坦的背景下观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦性首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的可怕山谷，在那里标签仍然被翻转。在广泛的实验中，我们在各种模型体系结构和数据集上观察到了这种现象，甚至对于经过恶意训练的模型也是如此。我们的结果也扩展到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。因此，理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健性模型的低全局Lipschitz常数相结合的必要性。



## **29. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

利用越狱概率攻击和保护多模式LLM cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 近年来，多通道大语言模型(MLLMS)在理解多通道内容方面表现出了优越的能力。然而，他们仍然容易受到越狱攻击，这些攻击利用他们的安全调整中的弱点来产生有害的反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLMS能力的二进制分类是不合适的。从这一观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表了当提示该输入时MLLMS生成恶意响应的可能性。我们通过对MLLMS的多次查询来近似这一概率。在使用越狱概率预测网络(JPPN)对输入隐藏状态与其对应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体地说，我们提出了基于越狱概率的攻击(JPA)，它优化了输入上的敌意扰动，以最大化越狱概率。为了对抗攻击，我们还提出了两种防御方法：基于越狱概率的精调(JPF)和基于越狱概率的防御噪声(JPDN)，它们分别在MLLM参数和输入空间中最小化越狱概率。大量的实验表明：(1)JPA在白盒和黑盒设置下都比以往的方法在小的扰动界和很少的迭代次数下都有很大的改善(高达28.38%)。(2)JPF和JPDN可显著减少越狱次数，最多可减少60%以上。以上两个结果都证明了引入越狱概率来细微区分不同输入越狱能力的意义。



## **30. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑匣子大型语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2310.12214v7) [paper-pdf](http://arxiv.org/pdf/2310.12214v7)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **31. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11934v3) [paper-pdf](http://arxiv.org/pdf/2412.11934v3)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **32. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

对大型语言模型进行水印可以阻止受版权保护的文本生成并隐藏训练数据吗？ cs.LG

19 pages, 7 figures. Published at AAAI 2025. Code will be available  at https://github.com/michael-panaitescu/watermark_copyright_aaai25

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.17417v2) [paper-pdf](http://arxiv.org/pdf/2407.17417v2)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. However, we also find that watermarking can have unintended consequences on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. These results reveal the complex interplay between different regulatory measures, which may impact each other in unforeseen ways. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

摘要: 大型语言模型(LLM)在生成丰富多样的文本方面表现出了令人印象深刻的能力。然而，由于LLMS可能无意中产生了受版权保护的材料，因此出现了对侵犯版权的担忧。在这篇文章中，我们首先研究了水印LLM作为对版权文本产生的威慑的有效性。通过理论分析和实证评估，我们证明了在LLMS中加入水印显著降低了产生受版权保护内容的可能性，从而解决了LLMS部署中的一个关键问题。然而，我们也发现，水印可能会对成员关系推断攻击(MIA)产生意想不到的后果，MIA旨在识别样本是否属于预训练数据集，并可能用于检测侵犯版权的行为。令人惊讶的是，我们发现水印对MIA的成功率产生了不利影响，使得在预训练数据集中检测受版权保护的文本的任务变得更加复杂。这些结果揭示了不同监管措施之间复杂的相互作用，这些措施可能会以不可预见的方式相互影响。最后，我们提出了一种自适应技术来提高最近的MIA在水印下的成功率。我们的发现强调了开发适应性方法来研究LLMS中具有潜在法律含义的关键问题的重要性。



## **33. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

CtrlRAG：检索增强语言生成中基于掩蔽语言模型的黑匣子对抗攻击 cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.

摘要: 检索-增强生成(RAG)系统通过集成外部知识库来增强大型语言模型(LLMS)。然而，这种集成带来了新的安全威胁：攻击者可以利用检索机制将恶意内容注入知识库，从而影响生成的响应。基于该攻击向量，我们提出了一种新的针对黑盒环境下RAG系统的攻击方法CtrlRAG，该方法符合实际场景。与现有的攻击方法不同，CtrlRAG引入了一种使用掩蔽语言模型(MLM)的扰动机制来动态优化恶意内容，以响应检索到的上下文的变化。实验结果表明，CtrlRAG在情绪操纵和幻觉放大目标上都优于三种基线方法。此外，我们评估了三种现有的防御机制，揭示了它们对CtrlRAG的有限有效性，并强调了对更强大防御的迫切需要。



## **34. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

Github: https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2411.13587v3) [paper-pdf](http://arxiv.org/pdf/2411.13587v3)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了两个非定向攻击目标，它们利用空间基础来破坏机器人动作的稳定性，以及一个操纵机器人轨迹的定向攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，我们促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前持续开发强大的防御策略的必要性。



## **35. Privacy Auditing of Large Language Models**

大型语言模型的隐私审计 cs.CR

ICLR 2025

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06808v1) [paper-pdf](http://arxiv.org/pdf/2503.06808v1)

**Authors**: Ashwinee Panda, Xinyu Tang, Milad Nasr, Christopher A. Choquette-Choo, Prateek Mittal

**Abstract**: Current techniques for privacy auditing of large language models (LLMs) have limited efficacy -- they rely on basic approaches to generate canaries which leads to weak membership inference attacks that in turn give loose lower bounds on the empirical privacy leakage. We develop canaries that are far more effective than those used in prior work under threat models that cover a range of realistic settings. We demonstrate through extensive experiments on multiple families of fine-tuned LLMs that our approach sets a new standard for detection of privacy leakage. For measuring the memorization rate of non-privately trained LLMs, our designed canaries surpass prior approaches. For example, on the Qwen2.5-0.5B model, our designed canaries achieve $49.6\%$ TPR at $1\%$ FPR, vastly surpassing the prior approach's $4.2\%$ TPR at $1\%$ FPR. Our method can be used to provide a privacy audit of $\varepsilon \approx 1$ for a model trained with theoretical $\varepsilon$ of 4. To the best of our knowledge, this is the first time that a privacy audit of LLM training has achieved nontrivial auditing success in the setting where the attacker cannot train shadow models, insert gradient canaries, or access the model at every iteration.

摘要: 目前的大型语言模型隐私审计技术的有效性有限--它们依赖于基本的方法来生成金丝雀，这导致了弱的成员关系推断攻击，进而给出了经验隐私泄露的松散下界。我们开发的金丝雀比以前在覆盖一系列现实环境的威胁模型下使用的金丝雀要有效得多。我们通过在多个微调LLM家族上的广泛实验证明，我们的方法为隐私泄露的检测设定了一个新的标准。在测量非私人训练的LLM的记忆率方面，我们设计的金丝雀超过了以前的方法。例如，在Qwen2.5-0.5B模型上，我们设计的金丝雀以1美元的FFP获得了49.6美元的TPR，大大超过了先前方法以1美元的FPR获得的4.2美元的TPR。我们的方法可以用来为理论上$varepsilon$为4训练的模型提供$\varepsilon\约1$的隐私审计。据我们所知，这是第一次在攻击者无法训练阴影模型、插入梯度金丝雀或在每次迭代中访问模型的情况下，LLM训练的隐私审计取得了非平凡的审计成功。



## **36. Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation**

小型语言模型能否可靠地抵抗越狱攻击？综合评价 cs.CR

19 pages, 12 figures

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06519v1) [paper-pdf](http://arxiv.org/pdf/2503.06519v1)

**Authors**: Wenhui Zhang, Huiyu Xu, Zhibo Wang, Zeqing He, Ziqi Zhu, Kui Ren

**Abstract**: Small language models (SLMs) have emerged as promising alternatives to large language models (LLMs) due to their low computational demands, enhanced privacy guarantees and comparable performance in specific domains through light-weight fine-tuning. Deploying SLMs on edge devices, such as smartphones and smart vehicles, has become a growing trend. However, the security implications of SLMs have received less attention than LLMs, particularly regarding jailbreak attacks, which is recognized as one of the top threats of LLMs by the OWASP. In this paper, we conduct the first large-scale empirical study of SLMs' vulnerabilities to jailbreak attacks. Through systematically evaluation on 63 SLMs from 15 mainstream SLM families against 8 state-of-the-art jailbreak methods, we demonstrate that 47.6% of evaluated SLMs show high susceptibility to jailbreak attacks (ASR > 40%) and 38.1% of them can not even resist direct harmful query (ASR > 50%). We further analyze the reasons behind the vulnerabilities and identify four key factors: model size, model architecture, training datasets and training techniques. Moreover, we assess the effectiveness of three prompt-level defense methods and find that none of them achieve perfect performance, with detection accuracy varying across different SLMs and attack methods. Notably, we point out that the inherent security awareness play a critical role in SLM security, and models with strong security awareness could timely terminate unsafe response with little reminder. Building upon the findings, we highlight the urgent need for security-by-design approaches in SLM development and provide valuable insights for building more trustworthy SLM ecosystem.

摘要: 小语言模型(SLM)作为大语言模型(LLM)的替代模型，具有较低的计算要求、增强的隐私保护以及通过轻量级的微调在特定领域具有相当的性能。在智能手机和智能汽车等边缘设备上部署SLM已成为一种日益增长的趋势。然而，小武器和轻武器的安全影响受到的关注较少，特别是在越狱攻击方面，而越狱攻击被认为是小岛屿发展中国家的最大威胁之一。在本文中，我们首次对SLM的越狱攻击脆弱性进行了大规模的实证研究。通过对来自15个主流SLM家庭的63个SLM与8种最新越狱方法的系统评估，我们发现47.6%的受评估SLM表现出对越狱攻击的高敏感性(ASR>40%)，其中38.1%的SLM甚至无法抵抗直接有害查询(ASR>50%)。我们进一步分析了漏洞背后的原因，并确定了四个关键因素：模型大小、模型体系结构、训练数据集和训练技术。此外，我们评估了三种提示级防御方法的有效性，发现它们都没有达到完美的性能，检测精度在不同的SLM和攻击方法中有所不同。值得注意的是，固有的安全意识在SLM安全中起着至关重要的作用，安全意识强的模型可以在几乎没有提醒的情况下及时终止不安全的响应。在这些发现的基础上，我们强调了在SLM开发中迫切需要通过设计来实现安全，并为建立更值得信赖的SLM生态系统提供了有价值的见解。



## **37. Life-Cycle Routing Vulnerabilities of LLM Router**

LLM路由器的生命周期路由漏洞 cs.CR

Work in progress

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.08704v1) [paper-pdf](http://arxiv.org/pdf/2503.08704v1)

**Authors**: Qiqi Lin, Xiaoyang Ji, Shengfang Zhai, Qingni Shen, Zhi Zhang, Yuejian Fang, Yansong Gao

**Abstract**: Large language models (LLMs) have achieved remarkable success in natural language processing, yet their performance and computational costs vary significantly. LLM routers play a crucial role in dynamically balancing these trade-offs. While previous studies have primarily focused on routing efficiency, security vulnerabilities throughout the entire LLM router life cycle, from training to inference, remain largely unexplored. In this paper, we present a comprehensive investigation into the life-cycle routing vulnerabilities of LLM routers. We evaluate both white-box and black-box adversarial robustness, as well as backdoor robustness, across several representative routing models under extensive experimental settings. Our experiments uncover several key findings: 1) Mainstream DNN-based routers tend to exhibit the weakest adversarial and backdoor robustness, largely due to their strong feature extraction capabilities that amplify vulnerabilities during both training and inference; 2) Training-free routers demonstrate the strongest robustness across different attack types, benefiting from the absence of learnable parameters that can be manipulated. These findings highlight critical security risks spanning the entire life cycle of LLM routers and provide insights for developing more robust models.

摘要: 大语言模型在自然语言处理方面取得了显著的成功，但它们的性能和计算代价差别很大。在动态平衡这些权衡方面，LLM路由器起着至关重要的作用。虽然以前的研究主要集中在路由效率上，但从培训到推断，整个LLM路由器生命周期的安全漏洞在很大程度上仍未被探索。本文对LLM路由器的生命周期路由漏洞进行了全面的研究。在广泛的实验设置下，我们通过几个典型的路由模型评估了白盒和黑盒对抗健壮性以及后门健壮性。我们的实验发现了几个关键发现：1)基于DNN的主流路由器往往表现出最弱的对抗性和后门健壮性，这主要是因为它们强大的特征提取能力，在训练和推理过程中放大了漏洞；2)无需训练的路由器在不同的攻击类型下表现出最强的健壮性，这得益于缺乏可学习的可操纵参数。这些发现突出了跨越LLM路由器整个生命周期的关键安全风险，并为开发更强大的模型提供了见解。



## **38. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

MM-PoisonRAG：通过本地和全球中毒攻击扰乱多模式RAG cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2502.17832v2) [paper-pdf](http://arxiv.org/pdf/2502.17832v2)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.

摘要: 配备了检索增强生成(RAG)的多通道大型语言模型(MLLMS)利用其丰富的参数知识和动态的外部知识来在问答等任务中脱颖而出。虽然RAG通过将响应建立在与查询相关的外部知识中来增强MLLS，但这种依赖构成了一个关键但未被开发的安全风险：知识中毒攻击，即故意将错误信息或无关知识注入外部知识库，以操纵不正确甚至有害的模型输出。为了暴露多模式RAG中的这些漏洞，我们提出了一种新的知识中毒攻击框架MM-PoisonRAG，该框架具有两种攻击策略：局部中毒攻击(LPA)和全局中毒攻击(GPA)，前者在文本和图像中注入特定于查询的错误信息以进行有针对性的操作，后者在MLLM生成过程中提供错误指导，以引发跨所有查询的无意义响应。我们跨多个任务、模型和访问设置评估我们的攻击，证明LPA成功操纵MLLM生成攻击者控制的答案，在多模式QA上的成功率高达56%。此外，GPA只需注入一次无关的知识，就能完全中断模型生成，准确率为0%。我们的结果突出表明，迫切需要针对知识中毒采取强有力的防御措施，以保护多式联运RAG框架。



## **39. Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions**

数据污染检测对LLM有效吗？检测假设的调查与评价 cs.CL

3 tables and 1 figures in the main text. This paper is accepted by  NAACL 2025 findings

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2410.18966v2) [paper-pdf](http://arxiv.org/pdf/2410.18966v2)

**Authors**: Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia

**Abstract**: Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.

摘要: 大型语言模型(LLM)在各种基准测试中表现出了出色的性能，显示出作为通用任务解算器的潜力。然而，由于LLM通常是根据大量数据进行培训的，其评估中的一个重大问题是数据污染，其中培训数据和评估数据集之间的重叠会夸大业绩评估。已经开发了多种方法来识别数据污染。这些方法依赖于特定的假设，而这些假设在不同的环境中可能并不普遍适用。为了弥补这一差距，我们系统地审查了50篇关于数据污染检测的论文，对潜在的假设进行了分类，并评估它们是否经过了严格的验证。我们识别和分析了八类假设，并测试了其中三类作为案例研究。我们的案例研究重点是检测直接的实例级数据污染，这也称为成员关系推断攻击(MIA)。我们的分析表明，基于这三个假设的MIA方法在LLM预训练中使用的数据集上具有类似于随机猜测的性能，这表明当前的LLM可能学习数据分布而不是记忆单个实例。同时，当可见实例和不可见实例之间存在数据分布变化时，MIA很容易失败。



## **40. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.00827v3) [paper-pdf](http://arxiv.org/pdf/2411.00827v3)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLBreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的稳健性--利用模型漏洞来引发有害输出的技术。然而，各种多模式数据的可获得性有限，限制了目前的方法严重依赖从有害文本数据集获得的对抗性或手动制作的图像，这些图像往往缺乏跨不同背景的有效性和多样性。在本文中，我们提出了一种新的越狱方法--IDEATOR，它能够自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样的见解，即VLM本身可以作为生成多模式越狱提示的强大红色团队模型。具体地说，Ideator利用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。大量的实验证明了IDAIDATER的高效率和可移植性，在越狱MiniGPT-4上达到了94%的攻击成功率(ASR)，平均只有5.34个查询，当转移到LLaVA、InstructBLIP和Chameleon时，ASR分别达到了82%、88%和75%。基于IDEATOR强大的可转移性和自动化流程，我们引入了VLBreakB边，这是一个包含3654个多模式越狱样本的安全基准。我们在最近发布的11个VLM上的基准结果显示，在安全对准方面存在显著差距。例如，我们的挑战集在GPT-40上达到了46.31%的ASR，在Claude-3.5-十四行诗上达到了19.65%，这突显了对更强大防御的迫切需要。



## **41. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

天花板：迈向视觉语言模型中经过认证的有毒感知距离 cs.CV

Under review of ICML 2025

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.10661v1) [paper-pdf](http://arxiv.org/pdf/2503.10661v1)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.

摘要: 大型视觉语言模型(VLM)的最新进展在广泛的视觉理解任务中显示出了显著的成功。然而，这些模型对越狱攻击的稳健性仍然是一个悬而未决的挑战。在这项工作中，我们提出了一个通用的认证防御框架，以严格保护VLM免受潜在的视觉越狱攻击。首先，我们提出了一种新的距离度量来量化恶意响应和预期响应之间的语义差异，该度量捕捉了传统的基于余弦相似性的度量经常忽略的细微差异。然后，我们设计了一种回归认证方法，该方法使用随机化平滑来提供形式上的健壮性保证，即使在黑盒设置下也不受对抗性和结构性扰动。作为补充，我们的特征空间防御将噪声分布(例如，高斯、拉普拉斯分布)引入到潜在嵌入中，以防止像素级和结构级的扰动。我们的结果突出了一种正式的、综合的战略的潜力，以建立更具弹性和更值得信赖的VLM。



## **42. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06269v1) [paper-pdf](http://arxiv.org/pdf/2503.06269v1)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 传统的白盒方法用于创建针对LLM的对抗性扰动，通常只依赖于目标模型的梯度计算，而忽略了攻击成败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新的白盒方法来弥合这一差距，该方法利用机械性的可解释性技术来制作实际的对抗性输入。具体地说，我们首先识别接受子空间-不会触发模型的拒绝机制的特征向量集合-然后使用基于梯度的优化将嵌入从拒绝子空间重定向到接受子空间，从而有效地实现越狱。这种有针对性的方法显著降低了计算成本，与通常失败或需要数小时计算的现有技术相比，在Gemma2、Llama3.2和Qwen2.5等最先进的模型上在几分钟甚至几秒钟内实现了80%-95%的攻击成功率。我们相信，这种方法为攻击研究和防御发展开辟了新的方向。此外，它还展示了机械可解释性在其他方法效率较低的情况下的实际应用，这突出了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting.上获得



## **43. Reinforced Diffuser for Red Teaming Large Vision-Language Models**

用于Red团队大型视觉语言模型的增强扩散器 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06223v1) [paper-pdf](http://arxiv.org/pdf/2503.06223v1)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The rapid advancement of large Vision-Language Models (VLMs) has raised significant safety concerns, particularly regarding their vulnerability to jailbreak attacks. While existing research primarily focuses on VLMs' susceptibility to harmful instructions, this work identifies a critical yet overlooked vulnerability: current alignment mechanisms often fail to address the risks posed by toxic text continuation tasks. To investigate this issue, we propose a novel Red Team Diffuser (RTD) framework, which leverages reinforcement learning to generate red team images that effectively induce highly toxic continuations from target black-box VLMs. The RTD pipeline begins with a greedy search for high-quality image prompts that maximize the toxicity of VLM-generated sentence continuations, guided by a Large Language Model (LLM). These prompts are then used as input for the reinforcement fine-tuning of a diffusion model, which employs toxicity and alignment rewards to further amplify harmful outputs. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% on the original attack set and 8.91% on a hold-out set. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. These findings reveal significant deficiencies in existing alignment strategies, particularly their inability to prevent harmful continuations. Our work underscores the urgent need for more robust and adaptive alignment mechanisms to ensure the safe deployment of VLMs in real-world applications.

摘要: 大型视觉语言模型(VLM)的快速发展引发了重大的安全问题，特别是它们对越狱攻击的脆弱性。虽然现有的研究主要集中在VLMS对有害指令的易感性上，但这项工作发现了一个严重但被忽视的漏洞：当前的比对机制往往无法解决有毒文本继续任务带来的风险。为了研究这一问题，我们提出了一种新的红色团队扩散器(RTD)框架，它利用强化学习来生成红色团队图像，从而有效地从目标黑盒VLM中诱导出剧毒的延续。RTD流水线从贪婪地搜索高质量的图像提示开始，这些图像提示在大型语言模型(LLM)的指导下，最大限度地提高了VLM生成的句子延续的毒性。这些提示随后被用作扩散模型的强化微调的输入，该模型使用毒性和配对奖励来进一步放大有害输出。实验结果证明了RTD的有效性，在原始攻击集和抵抗集上，LLaVA输出的毒力率分别提高了10.69%和8.91%。此外，RTD具有很强的跨模型传递性，对双子座和骆驼的毒性分别提高了5.1%和26.83%。这些调查结果揭示了现有调整战略的重大缺陷，特别是它们无法防止有害的延续。我们的工作强调了迫切需要更强大和适应性更强的对准机制，以确保在现实世界应用中安全地部署VLM。



## **44. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2410.02644v2) [paper-pdf](http://arxiv.org/pdf/2410.02644v2)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于大型语言模型(LLM)的代理可以使用外部工具和内存机制来解决复杂的现实任务，但它们也可能引入关键的安全漏洞。然而，现有的文献并没有全面评估对基于LLM的代理的攻击和防御。为了解决这一问题，我们引入了代理安全平台(ASB)，这是一个全面的框架，旨在形式化、基准和评估基于LLM的代理的攻击和防御，包括10个场景(例如，电子商务、自动驾驶、金融)、10个针对场景的代理、400多个工具、27种不同类型的攻击/防御方法和7个评估指标。在ASB的基础上，我们对10个即时注入攻击、一个内存中毒攻击、一个新的思维计划后门攻击、4个混合攻击以及13个LLM主干上的11个相应防御进行了基准测试。我们的测试结果揭示了代理操作的不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存恢复，平均攻击成功率最高为84.30\%，但现有防御措施的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的度量来评估代理在效用和安全性之间的平衡能力。我们的代码可以在https://github.com/agiresearch/ASB.上找到



## **45. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

您的基于LLM的文本到SQL模型安全吗？通过后门攻击探索SQL注入 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05445v1) [paper-pdf](http://arxiv.org/pdf/2503.05445v1)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型(LLM)在将自然语言问题转换为SQL查询(Text-to-SQL)方面取得了最先进的结果，这是数据库社区内的一个长期挑战。然而，安全问题在很大程度上仍然没有得到探索，特别是后门攻击的威胁，后门攻击可以通过微调有毒数据集将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的Text-to-SQL模型的漏洞，并提出了一种新的后门攻击框架ToxicSQL。我们的方法利用隐蔽的(语义和字符级触发器)使后门难以检测和删除，从而确保恶意行为保持隐蔽性，同时保持对良性输入的高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而能够生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中会带来严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就可以导致79.41%的攻击成功率，这对数据库安全构成了重大风险。此外，我们还提出了检测和缓解策略，以增强模型的可靠性。我们的发现强调了安全意识文本到SQL开发的迫切需要，强调了强大的后门威胁防御的重要性。



## **46. A Practical Memory Injection Attack against LLM Agents**

针对LLM代理的实用内存注入攻击 cs.LG

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.03704v2) [paper-pdf](http://arxiv.org/pdf/2503.03704v2)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.

摘要: 基于大型语言模型(LLM)的代理在广泛的复杂、真实世界的应用中表现出了强大的能力。然而，当检索用于演示的过去记录是恶意的时，具有受损内存库的LLM代理可能很容易产生有害输出。在本文中，我们提出了一种新的内存注入攻击，MinJA，它只需通过查询和输出观察与代理交互，就可以将恶意记录注入到内存库中。这些恶意记录旨在引发一系列恶意推理步骤，从而在执行受攻击用户的查询时导致不受欢迎的代理操作。具体地说，我们引入了一系列桥接步骤来将受害者查询与恶意推理步骤联系起来。在注入恶意记录的过程中，我们提出了一个指示提示，以引导代理自主生成我们设计的桥接步骤。我们还提出了一种渐进式缩短策略，逐步删除指示提示，以便在处理后续受害者查询时能够轻松检索到恶意记录。我们在不同代理上的广泛实验证明了Minja在损害代理内存方面的有效性。在执行要求最低的情况下，Minja使任何用户都能够影响代理内存，突出了LLM代理的实际风险。



## **47. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

双重后门：通过对抗性指令调优攻击将代码大型语言模型后门转换为传统恶意软件 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.

摘要: 为编码任务而设计的指令调整的大型语言模型越来越多地被用作人工智能编码助手。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。这项工作研究了将后门从AI/ML领域过渡到传统计算机恶意软件的新技术，揭示了人工智能和网络/软件安全的关键交集。为了探索这一交叉，我们提出了MalInstructCoder，一个旨在全面评估指令调优代码LLM的网络安全漏洞的框架。MalInstructCoder引入了自动数据中毒管道，将恶意代码片段注入良性代码，在保持功能有效性的同时毒化指令微调数据。它提出了两种具有实际安全含义的对抗性指令调整攻击：干净的即时中毒攻击和后门攻击。这些攻击旨在操纵Code LLM在特定攻击场景下生成包含恶意或有害功能的代码，同时保留预期功能。我们对代码特定指令调优过程的可利用性进行了全面的调查，涉及三个最先进的代码LLM：CodeLlama、DeepSeek-Coder和StarCoder2。我们的发现表明，这些模型非常容易受到我们的攻击。具体地说，干净的即时中毒攻击通过仅对指令微调数据集的1%(162个样本)下毒来实现从超过75%到86%的ASR@1。同样，后门攻击实现了从76%到86%的ASR@1，投毒率为0.5%。我们的研究揭示了指令调整的代码LLM构成的关键网络安全风险，并强调了对强大防御机制的迫切需要。



## **48. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

安全不仅仅是拒绝：推理增强微调，以实现可解释的LLM安全 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击利用了传统安全对齐中的弱点，传统安全对齐通常依赖僵化的拒绝启发式或表示工程来阻止有害输出。虽然它们对直接对抗性攻击是有效的，但它们不能满足更广泛的安全挑战，需要细致入微的、上下文感知的决策。为了解决这个问题，我们提出了针对可解释LLM安全(Rational)的推理增强精调，这是一个新的框架，它训练模型在响应之前进行显式的安全推理。微调模型利用自生成推理中丰富的预训练知识，通过结构化推理来引导自己的安全性，使上下文敏感决策内在化。我们的发现表明，安全超越了拒绝，需要背景感知才能做出更健壮、可解释和适应性更强的反应。推理是LLMS的核心能力，也是LLMS安全的基本机制。Rational采用了推理增强的微调，使其能够拒绝有害提示，同时在复杂场景中提供有意义的上下文感知响应。



## **49. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.06186v4) [paper-pdf](http://arxiv.org/pdf/2410.06186v4)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **50. Get my drift? Catching LLM Task Drift with Activation Deltas**

明白我的意思了吗？利用激活增量捕捉LLM任务漂移 cs.CR

SaTML 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2406.00799v6) [paper-pdf](http://arxiv.org/pdf/2406.00799v6)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: LLMs are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how users' tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and a suite of inspection tools.

摘要: LLM通常用于检索增强的应用程序中，以基于来自外部源的数据执行用户指令。例如，现代搜索引擎使用LLM根据相关搜索结果回答查询；电子邮件插件通过LLM处理电子邮件的内容来汇总电子邮件。然而，这些数据源的潜在不可信来源可能导致提示注入攻击，其中LLM被嵌入外部数据的自然语言指令操纵，导致其偏离用户的原始指令(S)。我们将这种偏差定义为任务漂移。任务漂移是一个重要的问题，因为它允许攻击者窃取数据或影响LLM对其他用户的输出。我们研究了LLM激活作为检测任务漂移的解决方案，表明激活增量-处理外部数据之前和之后的激活差异-与这一现象密切相关。通过两种探测方法，我们证明了一个简单的线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们通过对用户任务、系统提示和攻击的措辞做出最小假设来评估这些方法。我们观察到，这种方法对看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。有趣的是，该解决方案不需要对LLM进行任何修改(例如微调)，并且它与现有的元提示解决方案兼容，这使得它具有成本效益并且易于部署。为了鼓励对基于激活的任务检测、解码和可解释性的进一步研究，我们发布了我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自六个SOTA语言模型的表示，以及一套检测工具。



