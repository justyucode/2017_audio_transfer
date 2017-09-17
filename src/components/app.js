var React = require('react');
var ReactDOM = require('react-dom');
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import Load from './load.js';

class App extends React.Component {
    render () {
        return (
            <MuiThemeProvider>
                <Load/>
            </MuiThemeProvider>
        );
    }
};

ReactDOM.render(<App/>,  document.getElementById("app"));
