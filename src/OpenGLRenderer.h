#pragma once

#include <windows.h>
#include <GL/gl.h>
#include <string>

// Define missing OpenGL types and constants that are not in gl.h
#ifndef GL_VERSION_2_0
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
#endif

#ifndef GL_VERSION_3_0
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLubyte;
typedef float GLfloat;
typedef unsigned int GLenum;
typedef unsigned char GLboolean;
#endif

// OpenGL constants not in gl.h
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_ELEMENT_ARRAY_BUFFER
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80E1
#endif
#ifndef GL_TRIANGLES
#define GL_TRIANGLES 0x0004
#endif

class OpenGLRenderer {
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    
    bool Initialize(HWND hwnd, int width, int height);
    void Cleanup();
    
    bool Present(const uint8_t* data, int width, int height, int pitch);
    bool Resize(int width, int height);
    
    // Getters
    bool IsInitialized() const { return m_initialized; }
    
private:
    bool m_initialized;
    HWND m_hwnd;
    HDC m_hdc;
    HGLRC m_hrc;
    int m_width;
    int m_height;
    
    // OpenGL resources
    GLuint m_texture;
    GLuint m_program;
    GLuint m_vertexShader;
    GLuint m_fragmentShader;
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_ebo;
    
    // Shader uniforms
    GLint m_textureUniform;
    
    // Initialization helpers
    bool SetupOpenGL();
    bool LoadOpenGLExtensions();
    bool CreateShaders();
    bool CreateGeometry();
    bool CreateTexture();
    
    // OpenGL function pointers (for extensions)
    typedef void (WINAPI* PFNGLGENBUFFERSPROC)(GLsizei n, GLuint* buffers);
    typedef void (WINAPI* PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
    typedef void (WINAPI* PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
    typedef void (WINAPI* PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint* buffers);
    typedef void (WINAPI* PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint* arrays);
    typedef void (WINAPI* PFNGLBINDVERTEXARRAYPROC)(GLuint array);
    typedef void (WINAPI* PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint* arrays);
    typedef void (WINAPI* PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
    typedef void (WINAPI* PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
    typedef GLuint (WINAPI* PFNGLCREATESHADERPROC)(GLenum type);
    typedef void (WINAPI* PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
    typedef void (WINAPI* PFNGLCOMPILESHADERPROC)(GLuint shader);
    typedef void (WINAPI* PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint* params);
    typedef void (WINAPI* PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
    typedef void (WINAPI* PFNGLDELETESHADERPROC)(GLuint shader);
    typedef GLuint (WINAPI* PFNGLCREATEPROGRAMPROC)(void);
    typedef void (WINAPI* PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
    typedef void (WINAPI* PFNGLLINKPROGRAMPROC)(GLuint program);
    typedef void (WINAPI* PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint* params);
    typedef void (WINAPI* PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
    typedef void (WINAPI* PFNGLDELETEPROGRAMPROC)(GLuint program);
    typedef void (WINAPI* PFNGLUSEPROGRAMPROC)(GLuint program);
    typedef GLint (WINAPI* PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar* name);
    typedef void (WINAPI* PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
    typedef void (WINAPI* PFNGLACTIVETEXTUREPROC)(GLenum texture);
    
    // Function pointers
    PFNGLGENBUFFERSPROC glGenBuffers;
    PFNGLBINDBUFFERPROC glBindBuffer;
    PFNGLBUFFERDATAPROC glBufferData;
    PFNGLDELETEBUFFERSPROC glDeleteBuffers;
    PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
    PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
    PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
    PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
    PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
    PFNGLCREATESHADERPROC glCreateShader;
    PFNGLSHADERSOURCEPROC glShaderSource;
    PFNGLCOMPILESHADERPROC glCompileShader;
    PFNGLGETSHADERIVPROC glGetShaderiv;
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
    PFNGLDELETESHADERPROC glDeleteShader;
    PFNGLCREATEPROGRAMPROC glCreateProgram;
    PFNGLATTACHSHADERPROC glAttachShader;
    PFNGLLINKPROGRAMPROC glLinkProgram;
    PFNGLGETPROGRAMIVPROC glGetProgramiv;
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
    PFNGLDELETEPROGRAMPROC glDeleteProgram;
    PFNGLUSEPROGRAMPROC glUseProgram;
    PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
    PFNGLUNIFORM1IPROC glUniform1i;
    PFNGLACTIVETEXTUREPROC glActiveTexture;
    
    // Rendering helpers
    void SetupRenderState();
    void DrawQuad();
    
    void Reset();
};

// Vertex structure for full-screen quad
struct GLQuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};