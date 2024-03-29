<?php
namespace App\Http\Controllers;
use Illuminate\Http\Request;
use App\Helper\Reply;
use Response;
use App\Http\Requests\Media\StoreMedia;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\Input;

class MediaController extends Controller
{
    public function index() {
        $attachments_path = public_path('user-uploads/attachments/');
        $pdf_path = public_path('user-uploads/pdf/');
        $python_script = base_path('my_script.py');
        $file = '5cfad3d56a98e.jpg';
        $pdf = '5cfad3d56a98e.pdf';
        $cmd = 'python '.$python_script. ' "'.$attachments_path.$file.'" "'.$file.'" "'.$pdf_path.$pdf.'"';
        $output = shell_exec($cmd);
    }

    function uploadImage(Request $request){
        // Setup the validator
        $rules = array('image' => 'required|image|mimes:jpeg,png,jpg,gif|max:2048');
        $validator = Validator::make(Input::all(), $rules);

        // Validate the input and return correct response
        if ($validator->fails()){
          return Response::json(array(
              'success' => false,
              'errors' => $validator->getMessageBag()->toArray()

          ), 400); // 400 being the HTTP code for an invalid request.
        }

        /****** image ******/
        if ($request->hasFile('image')) {
          $extension = $request->file('image')->getClientOriginalExtension();
          $filename = uniqid();
          $file = $request->file('image')->move(config('constants.upload_path.attachments'), $filename.".".$extension);

          $attachments_path = public_path('user-uploads/attachments/');
          $pdf_path = public_path('user-uploads/pdf/');
          $file = $filename.".".$extension;
          $pdf = $filename.'.pdf';
          $python_script = base_path('media_upload.py');

          $cmd = 'python '.$python_script. ' "'.$attachments_path.$file.'" "'.$file.'" "'.$pdf_path.$pdf.'"';

          $output = shell_exec($cmd);
          if($output) {
              $pdf_file = config('constants.upload_path.pdf').$pdf;
              return Reply::dataOnly(['status' => 'success',
                  'file' => $file, 'pdf_file' => $pdf_file]);
          }
        }
    }
}
