import React from 'react';
import AppBar from 'material-ui/AppBar';
import Toggle from 'material-ui/Toggle';
import ReactAudioPlayer from 'react-audio-player';

 const styles = {
  block: {
    maxWidth: 250,
  },
  toggle: {
    marginBottom: 16,
  },
  thumbOff: {
    backgroundColor: '#ffcccc',
  },
  trackOff: {
    backgroundColor: '#ff9d9d',
  },
  thumbSwitched: {
    backgroundColor: 'red',
  },
  trackSwitched: {
    backgroundColor: '#ff9d9d',
  },
  labelStyle: {
    color: 'red',
  },
};


class Load extends React.Component {

  /* https://codepen.io/awesomecoding/pen/rVBaab audio visualization function below modified from open source code pen by Nick Jones
  */
  componentDidMount() {
    var file = document.getElementById("thefile");
    var audio = document.getElementById("audio");

    file.onchange = function() {
      var files = this.files;
      audio.src = URL.createObjectURL(files[0]);
      audio.load();
      audio.play();
      var context = new AudioContext();
      var src = context.createMediaElementSource(audio);
      var analyser = context.createAnalyser();

      var canvas = document.getElementById("canvas");
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight/2;
      var ctx = canvas.getContext("2d");

      src.connect(analyser);
      analyser.connect(context.destination);

      analyser.fftSize = 256;

      var bufferLength = analyser.frequencyBinCount;

      var dataArray = new Uint8Array(bufferLength);

      var WIDTH = canvas.width;
      var HEIGHT = canvas.height;

      var barWidth = (WIDTH / bufferLength) * 2.5;
      var barHeight;
      var x = 0;

      function renderFrame() {
        requestAnimationFrame(renderFrame);

        x = 0;

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = "#C5EFF7";
        ctx.fillRect(0, 0, WIDTH, HEIGHT);

        for (var i = 0; i < bufferLength; i++) {
          barHeight = dataArray[i];

          var r = barHeight + (25 * (i/bufferLength));
          var g = 250 * (i/bufferLength);
          var b = 50;

          ctx.fillStyle = "#F5D76E";
          ctx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);

          x += barWidth + 1;
        }
      }

      audio.play();
      renderFrame();
    };
  }

    render () {
      return <div>
        <AppBar
          title="StyleStream"
          iconClassNameRight="muidocs-icon-navigation-expand-more"
        />

        <div id="content">
        <input type="file" id="thefile" accept="audio/*" />
        <canvas id="canvas"></canvas>
        <audio id="audio" controls></audio>
        </div>
        <Toggle
            label="Toggled by default"
            style={styles.toggle}
            onToggle={() => console.log("wut")}
            defaultToggled={false}
          />
      </div>
    }
  }

export default Load;
