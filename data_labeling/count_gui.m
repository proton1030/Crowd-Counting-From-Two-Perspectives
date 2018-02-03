function varargout = count_gui(varargin)
% COUNT_GUI MATLAB code for count_gui.fig
%      COUNT_GUI, by itself, creates a new COUNT_GUI or raises the existing
%      singleton*.
%
%      H = COUNT_GUI returns the handle to a new COUNT_GUI or the handle to
%      the existing singleton*.
%
%      COUNT_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in COUNT_GUI.M with the given input arguments.
%
%      COUNT_GUI('Property','Value',...) creates a new COUNT_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before count_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to count_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help count_gui

% Last Modified by GUIDE v2.5 30-Nov-2017 01:15:49

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @count_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @count_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before count_gui is made visible.
function count_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to count_gui (see VARARGIN)

handles.last_pos = 0;
handles.crowd_cnt = zeros(10,1);
handles.axes1 = imshow(imread(sprintf('./3_L/%i.png', handles.last_pos)));
set(handles.edit1,'String',num2str(handles.last_pos));


% Choose default command line output for count_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% UIWAIT makes count_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);







% save('gt_3.mat');




% --- Outputs from this function are returned to the command line.
function varargout = count_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% load('gt_3.mat','handles');

if handles.last_pos ~= 0
    handles.last_pos = handles.last_pos - 1;
    handles.axes1 = imshow(imread(sprintf('./3_L/%i.png', handles.last_pos)));
end
set(handles.edit1,'String',num2str(handles.last_pos));
set(handles.edit2,'String',num2str(handles.crowd_cnt(handles.last_pos+1)));


handles.output = hObject;
guidata(hObject, handles);




% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% load('gt_3.mat','handles');
if handles.last_pos ~= 10
    handles.last_pos = handles.last_pos + 1;
    handles.axes1 = imshow(imread(sprintf('./3_L/%i.png', handles.last_pos)));
end
set(handles.edit1,'String',num2str(handles.last_pos));
set(handles.edit2,'String',num2str(handles.crowd_cnt(handles.last_pos+1)));


handles.output = hObject;
guidata(hObject, handles);



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double
handles.crowd_cnt(handles.last_pos+1:end) = str2double(get(hObject,'String'));
handles.output = hObject;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.crowd_cnt(handles.last_pos+1:end) = handles.crowd_cnt(handles.last_pos+1)-1;
set(handles.edit2,'String',num2str(handles.crowd_cnt(handles.last_pos+1)));
handles.output = hObject;
guidata(hObject, handles);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.crowd_cnt(handles.last_pos+1:end) = handles.crowd_cnt(handles.last_pos+1)+1;
set(handles.edit2,'String',num2str(handles.crowd_cnt(handles.last_pos+1)));
handles.output = hObject;
guidata(hObject, handles);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
crowd = handles.crowd_cnt;
save('gt.mat','crowd');
