# ALTER

## Abstract
![teaser](https://anonymous.4open.science/r/ALTER-72B0/figure/figure1.jpg)

## Dependencies
- python=3.9
- cudatoolkit=11.3
- torchvision=0.13.1
- pytorch=1.12.1
- torchaudio=0.12.1
- wandb=0.13.1
- scikit-learn=1.1.1
- pandas=1.4.3
- hydra-core=1.2.0

## Usage
<div>
  <input type="text" value="要复制的文本" id="copyText">
  <button onclick="copyToClipboard()">复制</button>
</div>

<script>
  function copyToClipboard() {
    var copyText = document.getElementById("copyText");
    copyText.select();
    document.execCommand("copy");
    alert("已复制: " + copyText.value);
  }
</script>
